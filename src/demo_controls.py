"""Display labels and presets only; pose signs and inference remain unchanged."""
PRESETS = {
    'Close & Level': [25., 0., 0., 0.],
    'Far & Level': [90., 0., 0., 0.],
    'Left & Above': [60., -15., 8., 0.],
    'Right & Below': [60., 15., -8., 0.],
    'Banked Formation': [60., 8., 3., 40.],
}
POSE_KEYS = ['scene_x','scene_y','scene_z','scene_roll']

def human_pose(pose, precision=1):
    x,y,z,roll = pose
    def directional(value,unit,negative,positive,zero):
        if abs(value) < .5*10**(-precision):
            return zero
        return f'{abs(value):.{precision}f}{unit} {negative if value < 0 else positive}'
    return [f'{x:.{precision}f} m ahead',
            directional(y,' m','left','right','Centered'),
            directional(z,' m','below','above','Same height'),
            directional(roll,'°','left bank','right bank','Level wings')]
