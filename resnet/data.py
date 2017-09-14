import os
import glob
import numpy as np
from PIL import Image

def path2np(path):
    img=Image.open(path)
    return np.asarray(img)

def save_paths(paths , save_path):
    f=open(save_path , 'w')
    for path in paths:
        f.write(path+'\n')
    f.close()

print os.getcwd()
def fundus_paths2imgs(source_folder='../../fundus_data/cropped_original_fundus_300x300' , normal_offset=100 ):
    print 'normal label : 0 , abnormal label : 1 '
    path, sub_folders , files=os.walk(source_folder).next()
    print sub_folders
    for i,sub_folder in enumerate(sub_folders):
        folder_path= os.path.join(path, sub_folder)
        folder_path= os.path.join(folder_path , '*.png')
        paths=glob.glob(folder_path)
        np_imgs=map(path2np , paths)
        if not sub_folders == 'normal':
            np_cls=np.ones(len(paths))
        else:
            np_cls = np.zeros(len(paths))
        target_folder=os.path.join(source_folder, sub_folder) #'../fundus_data/cropped_original_fundus_300x300/cataract '
        save_paths(paths , target_folder+'.txt') #'../fundus_data/cropped_original_fundus_300x300/cataract.txt '
        assert len(np_imgs) == len(paths)  == len(np_cls)
        print sub_folder + ':' + str(len(paths))

        if sub_folder == 'normal' and len(paths) >= normal_offset:
            n_normal=len(np_cls)
            share=n_normal/normal_offset
            for i in range(share):
                start = i*normal_offset
                end = (i+1)*normal_offset
                tmp_imgs=np_imgs[start:end]
                tmp_cls=np_cls[start:end]
                np.save(os.path.join(source_folder, sub_folder + '_'+str(i)+'_imgs'), tmp_imgs)
                np.save(os.path.join(source_folder, sub_folder + '_'+str(i)+'_labs'), tmp_cls)
        else:
            np.save(os.path.join(source_folder , sub_folder+'_imgs'), np_imgs)
            np.save(os.path.join(source_folder , sub_folder+'_labs'), np_cls)


def fundus_np_load(source_folder='../../fundus_data/cropped_original_fundus_300x300'):
    np_files = glob.glob(source_folder+'/*.npy')
    names=[]
    for np_file in np_files:
        path , f_name =os.path.split(np_file)
        f_name , ext =os.path.splitext(f_name)
        if '_img' in f_name:
            names.append(f_name.replace('_imgs', ''))
    names=list(set(names))

    def _fn(name):
        imgs=np.load(os.path.join(source_folder, name+'_imgs.npy' ))
        labs=np.load(os.path.join(source_folder, name+'_labs.npy' ))
        return imgs , labs
    for i,name in  enumerate(names):
        imgs ,labs=_fn(name)
        if i==0:
            ret_imgs=imgs
            ret_labs=labs
        else:
            ret_imgs=np.concatenate((ret_imgs , imgs ), axis=0)
            ret_labs = np.concatenate((ret_labs, labs), axis=0)
    print len(ret_imgs)
    print np.shape(ret_imgs)
    print len(ret_labs)
    print np.shape(ret_labs)
    return ret_imgs , ret_labs
    print names
if '__main__'== __name__:
    fundus_paths2imgs(source_folder='../../fundus_data/cropped_original_fundus_300x300' , normal_offset=15000 )


