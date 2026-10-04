"""Interactive views; camera detections decode the exact 42 ANN features."""
import numpy as np
import plotly.graph_objects as go
from utils import EDGES, NAMES, project
from demo_controls import human_pose

def formation_view(truth, prediction):
    """Top-view aircraft schematics, centered at the actual/predicted positions."""
    fig = go.Figure()
    # Icon offsets describe a simple airplane, not a second camera simulator.
    icon = np.array([[0,5],[.7,3],[.7,1],[5,-1],[5,-2],[.7,-1],
                     [.6,-3],[2,-4],[2,-4.7],[0,-4],[-2,-4.7],[-2,-4],
                     [-.6,-3],[-.7,-1],[-5,-2],[-5,-1],[-.7,1],[-.7,3],[0,5]])
    for pose,name,color,ghost in actors(truth,prediction):
        aircraft(fig,icon+[pose[1],pose[0]],name,color,ghost)
    fig.add_shape(type='line',x0=0,x1=0,y0=0,y1=truth[0],line=dict(color='#91a0ac',dash='dot'))
    fig.add_shape(type='line',x0=0,x1=truth[1],y0=truth[0],y1=truth[0],line=dict(color='#91a0ac',dash='dot'))
    fig.add_annotation(x=-9,y=truth[0]/2,text=human_pose(truth)[0],showarrow=False,textangle=-90)
    fig.add_annotation(x=truth[1],y=truth[0]+10,text=human_pose(truth)[1],showarrow=False)
    fig.add_annotation(x=0,y=-8,text='FOLLOWER',showarrow=False,font=dict(size=10))
    bound = max(29,abs(float(prediction[1]))+7)
    diagram_layout(fig,[-bound,bound],[-12,max(114,float(prediction[0])+12)],'LEFT ←   → RIGHT','AHEAD ↑')
    return fig

def actors(truth,prediction):
    return (([0,0,0,0],'Follower','#8cabbf',False),
            (truth,'Ground truth','#53b9dd',False),
            (prediction,'ANN prediction','#f5b66e',True))

def aircraft(fig,points,name,color,ghost):
    fig.add_trace(go.Scatter(x=points[:,0],y=points[:,1],mode='lines',name=name,
                            fill=None if ghost else 'toself',fillcolor=color,
                            line=dict(color=color,width=2,dash='dash' if ghost else 'solid'),
                            hovertemplate=name+'<extra></extra>'))

def diagram_layout(fig,xrange,yrange,xlabel,ylabel):
    fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=45),
                      xaxis=dict(range=xrange,title=xlabel,showgrid=False,showticklabels=False,zeroline=False,constrain='domain'),
                      yaxis=dict(range=yrange,title=ylabel,showgrid=False,showticklabels=False,zeroline=False,constrain='domain'),
                      legend=dict(orientation='h',y=-.2,font=dict(size=10)),hovermode='closest')

def side_view(truth,prediction):
    fig = go.Figure()
    icon = np.array([[5,0],[3,.6],[-1,.6],[-3,2],[-4,2],[-4,.3],[-5,0],[-4,-.5],
                     [-1,-.5],[1,-1],[3,-1],[1,-.3],[5,0]])
    for pose,name,color,ghost in actors(truth,prediction):
        aircraft(fig,icon+[pose[0],pose[2]],name,color,ghost)
    fig.add_shape(type='line',x0=0,x1=truth[0],y0=0,y1=0,line=dict(color='#91a0ac',dash='dot'))
    fig.add_shape(type='line',x0=truth[0],x1=truth[0],y0=0,y1=truth[2],line=dict(color='#91a0ac',dash='dot'))
    fig.add_annotation(x=truth[0],y=truth[2]+4,text=human_pose(truth)[2],showarrow=False)
    fig.add_annotation(x=truth[0]/2,y=-5,text=human_pose(truth)[0],showarrow=False)
    fig.add_annotation(x=0,y=-3,text='FOLLOWER',showarrow=False,font=dict(size=10))
    bound = max(16,abs(float(prediction[2]))+5)
    diagram_layout(fig,[-10,max(114,float(prediction[0])+12)],[-bound,bound],'AHEAD →','BELOW ← HEIGHT → ABOVE')
    return fig

def roll_view(truth,prediction):
    fig = go.Figure()
    # Frontal schematic rotates with the SAME Y/Z right-hand roll convention as project().
    icon = np.array([[-6,0],[-1,0],[-.5,.5],[0,1],[.5,.5],[1,0],[6,0],[6,-.35],
                     [1,-.35],[.5,-.8],[-.5,-.8],[-1,-.35],[-6,-.35],[-6,0]])
    fig.add_shape(type='line',x0=-7,x1=7,y0=0,y1=0,line=dict(color='#91a0ac',dash='dot'))
    for pose,name,color,ghost in actors(truth,prediction)[1:]:
        angle = np.deg2rad(pose[3])
        c,s = np.cos(angle),np.sin(angle)
        points = icon @ np.array([[c,s],[-s,c]])
        aircraft(fig,points,f'{name}: {human_pose(pose)[3]}',color,ghost)
    diagram_layout(fig,[-8,8],[-7,7],'View along the follower camera','')
    fig.update_yaxes(scaleanchor='x',scaleratio=1)
    return fig

def camera_view(features, config, truth=None, show_missing=False, full_frame=False):
    triples = np.asarray(features).reshape(14,3)
    visible = triples[:,2].astype(bool)
    pixels = (triples[:,:2]+1)*np.array([config['width'],config['height']])/2
    fig = go.Figure()
    x,y = [],[]
    # Additional fixed outline edges join only existing observed structural points.
    skeleton = EDGES + [(0,7),(0,10),(7,11),(10,12),(11,12),(11,13),(12,13)]
    for a,b in skeleton:
        if visible[a] and visible[b]:
            x.extend([pixels[a,0],pixels[b,0],None])
            y.extend([pixels[a,1],pixels[b,1],None])
    fig.add_trace(go.Scatter(x=x,y=y,mode='lines',line=dict(color='#8cabbf',width=3),
                            name='Detected structure',hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=pixels[visible,0],y=pixels[visible,1],mode='markers',name='Visible / detected',
                            marker=dict(color='#53b9dd',size=10,line=dict(color='#e2f4ff',width=1)),
                            text=[f'{i+1}: {NAMES[i]}' for i in np.flatnonzero(visible)],
                            hovertemplate='%{text}<br>u=%{x:.2f} px<br>v=%{y:.2f} px<extra></extra>'))
    if show_missing and truth is not None:
        ideal,in_frame = project(truth,config)
        missing = ~visible & in_frame
        fig.add_trace(go.Scatter(x=ideal[missing,0],y=ideal[missing,1],mode='markers',
                                name='Missing: truth overlay only',marker=dict(color='#f5b66e',symbol='x',size=9),
                                text=[NAMES[i] for i in np.flatnonzero(missing)],
                                hovertemplate='%{text}<br>Not supplied to ANN<extra></extra>'))
    xr,yr = [0,config['width']],[config['height'],0]
    if not full_frame:
        # Retain the optical center as a reference so offsets remain visually meaningful.
        shown = np.vstack([pixels[visible],[config['cx'],config['cy']]])
        if show_missing and truth is not None:
            shown = np.vstack([shown,ideal[missing]])
        lo,hi = shown.min(axis=0),shown.max(axis=0)
        center = (lo+hi)/2
        width = max(320,float(hi[0]-lo[0])*1.5,float(hi[1]-lo[1])*1.5*16/9)
        height = width*9/16
        xr = [center[0]-width/2,center[0]+width/2]
        yr = [center[1]+height/2,center[1]-height/2]
    fig.add_shape(type='line',x0=config['cx'],x1=config['cx'],y0=min(yr),y1=max(yr),
                  line=dict(color='#68798a',width=1,dash='dot'))
    fig.add_shape(type='line',x0=min(xr),x1=max(xr),y0=config['cy'],y1=config['cy'],
                  line=dict(color='#68798a',width=1,dash='dot'))
    fig.update_layout(height=440,margin=dict(l=15,r=15,t=15,b=15),
                      xaxis=dict(title='Camera horizontal position (pixels)',range=xr,constrain='domain'),
                      yaxis=dict(title='Camera vertical position (pixels)',range=yr,scaleanchor='x',scaleratio=1,constrain='domain'),
                      legend=dict(orientation='h',y=-.2),hovermode='closest')
    return fig

def loss_view(history):
    fig = go.Figure()
    for column,name,color in [('training_loss','Training','#53b9dd'),('validation_loss','Validation','#f5b66e')]:
        fig.add_trace(go.Scatter(x=history.epoch,y=history[column],mode='lines',name=name,line=dict(color=color)))
    fig.update_layout(height=350,margin=dict(l=15,r=15,t=15,b=15),
                      xaxis_title='Epoch',yaxis_title='MSE on standardized targets',
                      legend=dict(orientation='h'),hovermode='x unified')
    return fig
