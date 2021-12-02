import pandas as pd

# Enter path to the SEVIR data location
DATA_PATH= r"C:\Users\Owen\data" #path to location of sevir data on personal device
CATALOG_PATH = r'C:\Users\Owen\CATALOG.csv' #path to csv catalog of all sevir data on personal device, lets you reference stop instance specifically to obtain data

# On some Linux systems setting file locking to false is also necessary, excluded as personal system is windows so this is unecessary
#import os
#os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

#*****TUTORIAL EXAMPLE 1*****
import h5py  # needs conda/pip install h5py
import matplotlib.pyplot as plt

file_index = 0
with h5py.File( r'C:\Users\Owen\data\ir107\2019\SEVIR_IR107_RANDOMEVENTS_2019_0901_1231.h5', 'r') as hf: #path to spefic storm instance hdf5 file from sevir data on personal device
    event_id = hf['id'][file_index]
    vil = hf['ir107'][file_index] # must match the sensor type attatched to the event in line 15 or error, in this case ir107

print('Event ID:', event_id) #outputs the sevir event id # in console
print('Image shape:', vil.shape) #gives characteristics of images that will be displayed below

fig, axs = plt.subplots(1, 4, figsize=(10, 5))
axs[0].imshow(vil[:, :, 10])
axs[1].imshow(vil[:, :, 20])
axs[2].imshow(vil[:, :, 30])
axs[3].imshow(vil[:, :, 40])
plt.show() #shows the same area at 4 different time steps retrieved in lines 24-27, plot set up on line 23

#*****TUTORIAL EXAMPLE 2*****
# Read catalog
catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)

# Desired image types
img_types = set(['ir069','ir107','vil']) #outlines the desired image types an event must have to be considered in the set generated, vis omitted as didnt have a device with all 4 types on personal device

# Group by event id, and filter to only events that have all desired img_types
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id') #filters events based on requirements from line 35
event_ids = list(events.groups.keys())
print('Found %d events matching' % len(event_ids),img_types) #returns number of events which meet line 35 requirements
# writing the data into the file
import csv #defines csv
file = open('output1.csv', 'w+', newline ='') #defines what csv file to write into
with file:
    write = csv.writer(file)
    write.writerows(event_ids) #defines list to write into csv by rows,outputs list of event ids from event_ids in file

# Grab a sample event and view catalog entries
sample_event = events.get_group( event_ids[-1]) #pulls second to last event from event ids
print('Sample Event:',event_ids[-1]) #prints event id of sample event
df1=sample_event #defines sample event and its characteristics as data frame
df1.to_csv("output.csv", index=False) #outputs info on the sample event from line 49

#continuation of example 2, extract info on sample event
def read_data(sample_event, img_type, data_path=DATA_PATH): #def

    fn = sample_event[sample_event.img_type == img_type].squeeze().file_name #defines sample event name
    fi = sample_event[sample_event.img_type == img_type].squeeze().file_index #defines index value sample event is located at
    with h5py.File(data_path + '/' + fn, 'r') as hf: #here we grab the different images by keeping the path the same but itterating through the image type and appending that part of the path
        data = hf[img_type][fi]
    return data


#vis = read_data(sample_event, 'vis') <-the vis data for this storm instance wasnt downloaded so its omitted
ir069 = read_data(sample_event, 'ir069')
ir107 = read_data(sample_event, 'ir107')
vil = read_data(sample_event, 'vil')

# plot a frame from each img_type
fig, axs = plt.subplots(1, 3, figsize=(10, 5)) #sets up the plot/subplot sizings
frame_idx = 30
#axs[0].imshow(vis[:, :, frame_idx]), axs[0].set_title('VIS') <again the vis info is not present so its omitted
axs[0].imshow(ir069[:, :, frame_idx]), axs[0].set_title('IR 6.9') #note indexs were changes as only 3 images displayed here not 4
axs[1].imshow(ir107[:, :, frame_idx]), axs[1].set_title('IR 10.7')
axs[2].imshow(vil[:, :, frame_idx]), axs[2].set_title('VIL')
plt.show() #NOTE THE REASON VIS EXCLUDED IS THAT FILE DIDNT DOWNLOAD FOR THIS TIME INTERVAL

#start lightning tutorial
import numpy as np
def lght_to_grid(data): #defining the set up that will become the lighting info
    FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
    out_size = (48, 48, len(FRAME_TIMES))
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # filter out points outside the grid
    x, y = data[:, 3], data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[m, :]
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t = data[:, 0]
    z = np.digitize(t, FRAME_TIMES) - 1
    z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.float32)


def read_lght_data(sample_event, data_path=DATA_PATH):

    fn = sample_event[sample_event.img_type == 'lght'].squeeze().file_name #obtains/defines sample event name
    id = sample_event[sample_event.img_type == 'lght'].squeeze().id #obtains sample event id
    with h5py.File(data_path + '/' + fn, 'r') as hf: #obtains the light data for this storm instance using file path
        data = hf[id][:]
    return lght_to_grid(data)


lght = read_lght_data(sample_event)#defines lightning image for sample data

# include lightning counts in plot
fig, axs = plt.subplots(1, 4, figsize=(14, 5)) #defines plot and subplot sizing
frame_idx = 30
#axs[0].imshow(vis[:, :, frame_idx]), axs[0].set_title('VIS') <-againomitted as vis data for this istance not present
axs[0].imshow(ir069[:, :, frame_idx]), axs[0].set_title('IR 6.9') #like before indexs are changed from the tutorial  as theres no vis data
axs[1].imshow(ir107[:, :, frame_idx]), axs[1].set_title('IR 10.7')
axs[2].imshow(vil[:, :, frame_idx]), axs[2].set_title('VIL')
axs[3].imshow(lght[:, :, frame_idx]), axs[3].set_title('Lightning')
plt.show() #ouputs plot w/ subplots

# tutorial 3b
import sys
#enables color map change
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap #using different color schemes

def get_cmap(type, encoded=True):
    if type.lower() == 'vis':
        cmap, norm = vis_cmap(encoded)
        vmin, vmax = (0, 10000) if encoded else (0, 1)
    elif type.lower() == 'vil':
        cmap, norm = vil_cmap(encoded)
        vmin, vmax = None, None
    elif type.lower() == 'ir069':
        cmap, norm = c09_cmap(encoded)
        vmin, vmax = (-8000, -1000) if encoded else (-80, -10)
    elif type.lower() == 'lght':
        cmap, norm = 'hot', None
        vmin, vmax = 0, 5
    else:
        cmap, norm = 'jet', None
        vmin, vmax = (-7000, 2000) if encoded else (-70, 20)

    return cmap, norm, vmin, vmax


def vil_cmap(encoded=True): #assigns values for vil data
    cols = [[0, 0, 0],
            [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
            [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
            [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
            [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
            [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
            [0.9607843137254902, 0.9607843137254902, 0.0],
            [0.9294117647058824, 0.6745098039215687, 0.0],
            [0.9411764705882353, 0.43137254901960786, 0.0],
            [0.6274509803921569, 0.0, 0.0],
            [0.9058823529411765, 0.0, 1.0]]
    lev = [16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255]
    # TODO:  encoded=False
    nil = cols.pop(0) #starts to assign colors based on definitions from above
    under = cols[0]
    over = cols.pop()
    cmap = mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap, norm


def vis_cmap(encoded=True):#recolors vis data set, defines vis data here
    cols = [[0, 0, 0],
            [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
            [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
            [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
            [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
            [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
            [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
            [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
            [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
            [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
            [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
            [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
            [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
            [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
            [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
            [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
            [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
            [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
            [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
            [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
            [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
            [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
            [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
            [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
            [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
            [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
            [0.9803921568627451, 0.9803921568627451, 0.9803921568627451]]
    lev = np.array([0., 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2, 0.24,
                    0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68,
                    0.72, 0.76, 0.8, 0.9, 1.])
    if encoded:
        lev *= 1e4
    nil = cols[0]
    under = cols[0]
    over = cols.pop()
    cmap = mpl.colors.ListedColormap(cols) # actually assigns colors here
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap, norm


def ir_cmap(encoded=True):#defines ir data
    cols = [[0, 0, 0], [1.0, 1.0, 1.0],
            [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
            [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
            [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
            [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
            [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
            [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
            [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
            [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
            [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
            [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
            [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
            [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
            [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
            [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
            [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
            [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
            [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
            [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
            [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
            [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
            [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
            [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
            [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
            [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
            [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
            [0.0, 0.803921568627451, 0.803921568627451]]
    lev = np.array([-110., -105.2, -95.2, -85.2, -75.2, -65.2, -55.2, -45.2,
                    -35.2, -28.2, -23.2, -18.2, -13.2, -8.2, -3.2, 1.8,
                    6.8, 11.8, 16.8, 21.8, 26.8, 31.8, 36.8, 41.8,
                    46.8, 51.8, 90., 100.])
    if encoded:
        lev *= 1e2
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap = mpl.colors.ListedColormap(cols)#recolros based on definitions shown above
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap, norm


def c09_cmap(encoded=True):
    cols = [
        [1.000000, 0.000000, 0.000000],
        [1.000000, 0.031373, 0.000000],
        [1.000000, 0.062745, 0.000000],
        [1.000000, 0.094118, 0.000000],
        [1.000000, 0.125490, 0.000000],
        [1.000000, 0.156863, 0.000000],
        [1.000000, 0.188235, 0.000000],
        [1.000000, 0.219608, 0.000000],
        [1.000000, 0.250980, 0.000000],
        [1.000000, 0.282353, 0.000000],
        [1.000000, 0.313725, 0.000000],
        [1.000000, 0.349020, 0.003922],
        [1.000000, 0.380392, 0.003922],
        [1.000000, 0.411765, 0.003922],
        [1.000000, 0.443137, 0.003922],
        [1.000000, 0.474510, 0.003922],
        [1.000000, 0.505882, 0.003922],
        [1.000000, 0.537255, 0.003922],
        [1.000000, 0.568627, 0.003922],
        [1.000000, 0.600000, 0.003922],
        [1.000000, 0.631373, 0.003922],
        [1.000000, 0.666667, 0.007843],
        [1.000000, 0.698039, 0.007843],
        [1.000000, 0.729412, 0.007843],
        [1.000000, 0.760784, 0.007843],
        [1.000000, 0.792157, 0.007843],
        [1.000000, 0.823529, 0.007843],
        [1.000000, 0.854902, 0.007843],
        [1.000000, 0.886275, 0.007843],
        [1.000000, 0.917647, 0.007843],
        [1.000000, 0.949020, 0.007843],
        [1.000000, 0.984314, 0.011765],
        [0.968627, 0.952941, 0.031373],
        [0.937255, 0.921569, 0.050980],
        [0.901961, 0.886275, 0.074510],
        [0.870588, 0.854902, 0.094118],
        [0.835294, 0.823529, 0.117647],
        [0.803922, 0.788235, 0.137255],
        [0.772549, 0.756863, 0.160784],
        [0.737255, 0.725490, 0.180392],
        [0.705882, 0.690196, 0.200000],
        [0.670588, 0.658824, 0.223529],
        [0.639216, 0.623529, 0.243137],
        [0.607843, 0.592157, 0.266667],
        [0.572549, 0.560784, 0.286275],
        [0.541176, 0.525490, 0.309804],
        [0.509804, 0.494118, 0.329412],
        [0.474510, 0.462745, 0.349020],
        [0.752941, 0.749020, 0.909804],
        [0.800000, 0.800000, 0.929412],
        [0.850980, 0.847059, 0.945098],
        [0.898039, 0.898039, 0.964706],
        [0.949020, 0.949020, 0.980392],
        [1.000000, 1.000000, 1.000000],
        [0.964706, 0.980392, 0.964706],
        [0.929412, 0.960784, 0.929412],
        [0.890196, 0.937255, 0.890196],
        [0.854902, 0.917647, 0.854902],
        [0.815686, 0.894118, 0.815686],
        [0.780392, 0.874510, 0.780392],
        [0.745098, 0.850980, 0.745098],
        [0.705882, 0.831373, 0.705882],
        [0.670588, 0.807843, 0.670588],
        [0.631373, 0.788235, 0.631373],
        [0.596078, 0.764706, 0.596078],
        [0.560784, 0.745098, 0.560784],
        [0.521569, 0.721569, 0.521569],
        [0.486275, 0.701961, 0.486275],
        [0.447059, 0.678431, 0.447059],
        [0.411765, 0.658824, 0.411765],
        [0.376471, 0.635294, 0.376471],
        [0.337255, 0.615686, 0.337255],
        [0.301961, 0.592157, 0.301961],
        [0.262745, 0.572549, 0.262745],
        [0.227451, 0.549020, 0.227451],
        [0.192157, 0.529412, 0.192157],
        [0.152941, 0.505882, 0.152941],
        [0.117647, 0.486275, 0.117647],
        [0.078431, 0.462745, 0.078431],
        [0.043137, 0.443137, 0.043137],
        [0.003922, 0.419608, 0.003922],
        [0.003922, 0.431373, 0.027451],
        [0.003922, 0.447059, 0.054902],
        [0.003922, 0.462745, 0.082353],
        [0.003922, 0.478431, 0.109804],
        [0.003922, 0.494118, 0.137255],
        [0.003922, 0.509804, 0.164706],
        [0.003922, 0.525490, 0.192157],
        [0.003922, 0.541176, 0.215686],
        [0.003922, 0.556863, 0.243137],
        [0.007843, 0.568627, 0.270588],
        [0.007843, 0.584314, 0.298039],
        [0.007843, 0.600000, 0.325490],
        [0.007843, 0.615686, 0.352941],
        [0.007843, 0.631373, 0.380392],
        [0.007843, 0.647059, 0.403922],
        [0.007843, 0.662745, 0.431373],
        [0.007843, 0.678431, 0.458824],
        [0.007843, 0.694118, 0.486275],
        [0.011765, 0.705882, 0.513725],
        [0.011765, 0.721569, 0.541176],
        [0.011765, 0.737255, 0.568627],
        [0.011765, 0.752941, 0.596078],
        [0.011765, 0.768627, 0.619608],
        [0.011765, 0.784314, 0.647059],
        [0.011765, 0.800000, 0.674510],
        [0.011765, 0.815686, 0.701961],
        [0.011765, 0.831373, 0.729412],
        [0.015686, 0.843137, 0.756863],
        [0.015686, 0.858824, 0.784314],
        [0.015686, 0.874510, 0.807843],
        [0.015686, 0.890196, 0.835294],
        [0.015686, 0.905882, 0.862745],
        [0.015686, 0.921569, 0.890196],
        [0.015686, 0.937255, 0.917647],
        [0.015686, 0.952941, 0.945098],
        [0.015686, 0.968627, 0.972549],
        [1.000000, 1.000000, 1.000000]]

    return ListedColormap(cols), None

import sys
sys.path.append(r"C:\Users\Owen\data\eie-sevir-master\eie-sevir-master\storm_events") # add sevir module to path
#from sevir.display import get_cmap
# Get colormaps for encoded types
vis_cmap,vis_norm,vis_vmin,vis_vmax = get_cmap('vis',encoded=True)
ir069_cmap,ir069_norm,ir069_vmin,ir069_vmax = get_cmap('ir069',encoded=True)
ir107_cmap,ir107_norm,ir107_vmin,ir107_vmax = get_cmap('ir107',encoded=True)
vil_cmap,vil_norm,vil_vmin,vil_vmax = get_cmap('vil',encoded=True)
lght_cmap,lght_norm,lght_vmin,lght_vmax = get_cmap('lght',encoded=True)

fig,axs = plt.subplots(1,4,figsize=(14,5))
frame_idx = 30
#axs[0].imshow(vis[:,:,frame_idx],cmap=vis_cmap,norm=vis_norm,vmin=vis_vmin,vmax=vis_vmax), axs[0].set_title('VIS') <-omitted as no vis data for this instance
axs[0].imshow(ir069[:,:,frame_idx],cmap=ir069_cmap,norm=ir069_norm,vmin=ir069_vmin,vmax=ir069_vmax), axs[0].set_title('IR 6.9') #again indexs changed as there is no vis data here
axs[1].imshow(ir107[:,:,frame_idx],cmap=ir107_cmap,norm=ir107_norm,vmin=ir107_vmin,vmax=ir107_vmax), axs[1].set_title('IR 10.7')
axs[2].imshow(vil[:,:,frame_idx],cmap=vil_cmap,norm=vil_norm,vmin=vil_vmin,vmax=vil_vmax), axs[2].set_title('VIL')
axs[3].imshow(lght[:,:,frame_idx],cmap=lght_cmap,norm=lght_norm,vmin=lght_vmin,vmax=lght_vmax), axs[3].set_title('Lightning')
plt.show() #shows sub plots

#tutorial 4 geo referencing
import re
import numpy as np
class LaeaProjection(): #defines how to do laeaprojection as a class
    def __init__(self,event):
        self.proj = event.proj
        self.lat0 = float(re.compile('\+lat_0=([+-]?\d+)').search(self.proj).groups()[0])
        self.lon0 = float(re.compile('\+lon_0=([+-]?\d+)').search(self.proj).groups()[0])
        self.R = float(re.compile('\+a=(\d+)').search(self.proj).groups()[0])
        self.llcrnlat = event.llcrnrlat
        self.llcrnlon = event.llcrnrlon
        self.refX, self.refY = self.forward(self.llcrnlon,self.llcrnlat,pixel=False)
        self.binX = event.width_m / event.size_x
        self.binY = event.height_m / event.size_y

    def forward(self,lon,lat,pixel=True):
        """
        Maps lat/lon to pixel x,y.  For projection coordinates instead of pixel, set pixel=False.
        """
        sind = lambda t: np.sin(t*np.pi/180)
        cosd = lambda t: np.cos(t*np.pi/180)
        k = self.R * np.sqrt(2/(1+sind(self.lat0)*sind(lat)+cosd(self.lat0)*cosd(lat)*cosd(lon-self.lon0)))
        x = k*cosd(lat)*sind(lon-self.lon0)
        y = k*(cosd(self.lat0)*sind(lat) - sind(self.lat0)*cosd(lat)*cosd(lon-self.lon0))
        if pixel:
            x = (x-self.refX) / self.binX
            y = (y-self.refY) / self.binY
        return x,y
    def inverse(self,x,y,pixel=True):
        """
        Maps pixel coordinates to (lon,lat) position.  If passing projection corrdinates, set pixel=False.
        """
        if pixel:
            x = x*self.binX + self.refX
            y = y*self.binY + self.refY
        x/=self.R
        y/=self.R
        sind = lambda t: np.sin(t*np.pi/180)
        cosd = lambda t: np.cos(t*np.pi/180)
        rho = np.sqrt(x*x+y*y)
        c = 2*np.arcsin(0.5*rho)
        sinc = np.sin(c)
        cosc = np.cos(c)
        lat = 180/np.pi*np.arcsin(cosc*sind(self.lat0)+y*sinc*cosd(self.lat0)/rho)
        lon = self.lon0+180/np.pi*np.arctan(x*sinc/(rho*cosd(self.lat0)*cosc - y*sind(self.lat0)*sinc))
        return lon,lat

#start geo referencing section
proj=LaeaProjection( sample_event[sample_event.img_type=='vil'].squeeze() )
X, Y = np.meshgrid(np.arange(vil.shape[0]), np.arange(vil.shape[0]))
lons, lats = proj.inverse(X, Y)

# Plot with origin='lower' so up corresponds to north.
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(vil[:, :, 20], origin='lower')
la = ax[1].imshow(lats, origin='lower')
fig.colorbar(la, ax=ax[1])
ax[1].set_title('Pixel Latitudes')
lo = ax[2].imshow(lons, origin='lower')
fig.colorbar(lo, ax=ax[2])
ax[2].set_title('Pixel Longitudes')
plt.show()

#to include marker at specific point , duluth mn
lat,lon = 46.7867, -92.1005 # Duluth, MN
x,y=proj.forward(lon,lat)
print('x=%f,y=%f' % (x,y))
# Plot with origin='lower' so up corresponds to north.
fig,ax=plt.subplots(1,1,figsize=(5,5))
ax.imshow(vil[:,:,20],origin='lower')
ax.plot(x,y,linestyle='none', marker="o", markersize=16, alpha=0.6, c="red")
ax.text(x-30,y-30,'Duluth, MN',color='r')
plt.show()

# Note:  Requires basemap module to run
import warnings
warnings.filterwarnings('ignore')
import cartopy
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature

fig = plt.figure(figsize=(8, 8))
# Determine Extents from sample event
s=sample_event[sample_event.img_type=='vil'].squeeze()
img_extent = [s.llcrnrlon, s.urcrnrlon,s.urcrnrlat,s.llcrnrlat]
# Calculate central lat long, Set up plot Region
central_longitude=(s.urcrnrlon+s.llcrnrlon) / 2
central_latitude=(s.urcrnrlat+s.llcrnrlat) / 2
ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_longitude, central_latitude=central_latitude), extent=img_extent)
# Add Cartopy Features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'),edgecolor='black')
ax.add_feature(cartopy.feature.STATES.with_scale('10m'),edgecolor='black')
ax.add_feature(cartopy.feature.LAKES.with_scale('10m'),edgecolor='black', facecolor='none')
# Add Derived Image
ax.imshow(vil[:, :, 24], extent=img_extent, transform=ccrs.PlateCarree())
# Add Duluth Marker
dlong, dlat = -92.1005, 46.7867
ax.plot(dlong, dlat, 'ro', markersize=16, transform=ccrs.Geodetic())
ax.text(dlong, dlat, 'Duluth, MN', transform=ccrs.Geodetic())

plt.show()





