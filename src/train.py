import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from utils import ROOT, FEATURES, TARGETS, seed_all, save_json
from preprocessing import prepare, dataset_hash
from model import PoseANN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--patience',type=int,default=40)
    parser.add_argument('--smoke',action='store_true',help='Write smoke artifacts separately; do not replace the final model.')
    args = parser.parse_args()
    seed_all()
    frame,indices,scaler = prepare()
    tensors = {}
    for name in ('train','validation'):
        rows = frame.iloc[indices[name]]
        tensors[name] = (torch.tensor(rows[FEATURES].to_numpy(),dtype=torch.float32),
                         torch.tensor(scaler.transform(rows[TARGETS]),dtype=torch.float32))
    loader = DataLoader(TensorDataset(*tensors['train']),batch_size=args.batch_size,shuffle=True)
    model = PoseANN()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.MSELoss()
    best, stale, history, best_epoch = float('inf'),0,[],0
    suffix = '_smoke' if args.smoke else ''
    start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        total = 0.
        for x,y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x),y)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x)
        model.eval()
        with torch.no_grad():
            val = criterion(model(tensors['validation'][0]),tensors['validation'][1]).item()
        history.append(dict(epoch=epoch,training_loss=total/len(indices['train']),validation_loss=val))
        if val < best:
            best,stale,best_epoch = val,0,epoch
            torch.save(dict(state_dict=model.state_dict(),dataset_sha256=dataset_hash(),best_epoch=epoch,
                            architecture=[42,100,100,4],feature_columns=FEATURES,target_columns=TARGETS),
                       ROOT/f'models/best_model{suffix}.pt')
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f'Epoch {epoch}: train={history[-1]["training_loss"]:.6f}, validation={val:.6f}',flush=True)
        if stale >= args.patience:
            break
    hist = pd.DataFrame(history)
    hist.to_csv(ROOT/f'results/training_history{suffix}.csv',index=False)
    save_json(ROOT/f'results/training_summary{suffix}.json',dict(seed=42,optimizer='Adam',learning_rate=args.lr,
              batch_size=args.batch_size,max_epochs=args.epochs,epochs_completed=len(hist),patience=args.patience,
              best_epoch=best_epoch,best_validation_loss=best,seconds=time.perf_counter()-start))
    fig,ax = plt.subplots(figsize=(8,4.5))
    ax.plot(hist.epoch,hist.training_loss,label='Training')
    ax.plot(hist.epoch,hist.validation_loss,label='Validation')
    ax.set(xlabel='Epoch',ylabel='MSE on standardized targets',title='ANN training and validation loss')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT/f'results/training_validation_loss{suffix}.png',dpi=160)
    plt.close(fig)
    print(f'Best epoch: {best_epoch}; validation loss: {best:.6f}',flush=True)

if __name__ == '__main__':
    main()
