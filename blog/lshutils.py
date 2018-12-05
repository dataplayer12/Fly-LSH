from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf
import numpy as np
import pickle
import os
from bokeh.plotting import figure,output_file,show

class Dataset(object):
    def __init__(self,name,path='./datasets/'):
        self.path=path
        self.name=name.upper()
        if self.name=='MNIST' or self.name=='FMNIST':
            self.indim=784
            try:
                self.data=read_data_sets(self.path+self.name)
            except OSError as err:
                print(str(err))
                raise ValueError('Try again')

        elif self.name=='CIFAR10':
            self.indim=(32,32,3)
            if self.name not in os.listdir(self.path):
                print('Data not in path')
                raise ValueError()
        elif self.name=='GLOVE':
            self.indim=300
            self.data=pickle.load(open(self.path+'glove30k.p','rb'))

        elif self.name=='SIFT': #SIFT features dataset
            self.indim=128
            self.data=loadmat(self.path+self.name+'/siftvecs.mat')['vecs']

        elif self.name=='GIST': #GIST dataset
            self.indim=960
            self.data=loadmat(self.path+self.name+'gistvecs.mat')['vecs']

        elif self.name=='LMGIST': #LabelMe dataset
            self.indim=512
            self.data=loadmat(self.path+self.name+'/LabelMe_gist.mat')['gist']

        elif self.name=='RANDOM':
            self.indim=128
            self.data=np.random.random(size=(100_000,self.indim)) #np.random.randn(100_000,self.indim)

class LSH(object):
    def __init__(self,data,hash_length):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        """
        self.data=data-np.mean(data,axis=1)[:,None]
        self.weights=np.random.random((data.shape[1],hash_length))
        self.hashes=(self.data@self.weights)>0
    
    def query(self,qidx,nnn):
        L1_distances=np.sum(np.abs(self.hashes[qidx,:]^self.hashes),axis=1)
        NNs=L1_distances.argsort()[1:nnn+1]
        #print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs

    def true_nns(self,qidx,nnn):
        sample=self.data[qidx,:]
        return np.sum((self.data-sample)**2,axis=1).argsort()[1:nnn+1]
        
    def construct_true_nns(self,indices,nnn):
        all_NNs=np.zeros((len(indices),nnn))
        for idx1,idx2 in enumerate(indices):
            all_NNs[idx1,:]=self.true_nns(idx2,nnn)
        return all_NNs
    
    def AP(self,predictions,truth):
        assert len(predictions)==len(truth)
        precisions=[len(list(set(predictions[:idx]).intersection(set(truth[:idx]))))/idx for\
                    idx in range(1,len(truth)+1)]
        return np.mean(precisions)
    
    def findmAP(self,nnn,n_points):
        start=np.random.randint(low=0,high=self.data.shape[0]-n_points)
        sample_indices=np.arange(start,start+n_points)
        all_NNs=self.construct_true_nns(sample_indices,nnn)
        self.allAPs=[]
        for eidx,didx in enumerate(sample_indices):
            #eidx: enumeration id, didx: index of sample in self.data
            this_nns=self.query(didx,nnn)
            this_AP=self.AP(list(this_nns),list(all_NNs[eidx,:]))
            #print(this_AP)
            self.allAPs.append(this_AP)
            
        return np.mean(self.allAPs)


class flylsh(LSH):
    def __init__(self,data,hash_length,sampling_ratio,embedding_size):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.embedding_size=embedding_size
        self.data=(data-np.mean(data,axis=1)[:,None])
        weights=np.random.random((data.shape[1],embedding_size))
        self.weights=(weights>1-sampling_ratio) #sparse projection vectors
        all_activations=(self.data@self.weights)
        threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        #print(threshold[0])
        self.hashes=(all_activations>=threshold) #choose topk activations
        #print(self.hashes.sum(axis=1)[0])
        #for idx in range(data.shape[0]):
        #    self.hashes[idx,top_activations[idx,:]]=True


class AEflylsh(LSH): 
#implements Fly LSH where weights are pre-specified
#The weights passed to init should be learnt from an autoencoder
    def __init__(self,data,hash_length,sampling_ratio,weights):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        weights: weights learnt from an autoencoder
        weights should have the same dimensionality as projection space (m)
        """
        #assert weights.shape[1]==embedding_size, f'Expects a {embedding_size} dim embedding from {weights.shhape[1]} dim weights'
        self.embedding_size=weights.shape[1]
        self.data=(data-np.mean(data,axis=1)[:,None])
        self.weights=(weights>=np.sort(weights,axis=0)[-int(weights.shape[0]*sampling_ratio),:][None,:]) #sparse projection vectors
        all_activations=(self.data@self.weights)
        threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        self.hashes=(all_activations>=threshold) #choose top k activations

class AutoEncoder(object):
    def __init__(self,nodes,is_sparse=False,rho=0.1,beta=2,dec_weights=None):
        """
        nodes: a list [in_dim,n_hidden]
        is_sparse: bool
        rho: sparsity factor (fraction of weights turned on)
        beta: weight of kl_divergence loss
        total_loss=reconstruction_loss+beta*kl_divergence
        """
        self.in_dim=nodes[0]
        self.n_hidden=nodes[1]
        self.epochs=5
        self.learn_rate=[1e-3/(2**(e//3)) for e in range(self.epochs)]
        self.batch_size=32
        self.inputs_,self.targets,self.lr=self.get_placeholders()
        if is_sparse:
            self.rho=rho
            self.encode_weights=tf.Variable(tf.random_uniform([self.in_dim,self.n_hidden],minval=0,maxval=10*self.rho))
            self.decode_weights=tf.Variable(tf.truncated_normal([self.in_dim,self.n_hidden],stddev=0.05))
        else:
            self.encode_weights=tf.Variable(tf.truncated_normal([self.in_dim,self.n_hidden],stddev=0.05))
            self.decode_weights=self.encode_weights

        #biases=tf.Variable(tf.zeros([self.n_hidden])) #we don't want to use biases
        hlayer=tf.matmul(self.inputs_,self.encode_weights)
        self.hlayer=tf.nn.relu(hlayer)
        output=tf.matmul(self.hlayer,tf.transpose(self.decode_weights))
        self.output=tf.nn.sigmoid(output)
        self.recon_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets,logits=self.output)) #LAST CHANGE HERE
        normed= lambda w: (w-tf.reduce_min(w))/(tf.reduce_max(w)-tf.reduce_min(w))
        #normalize things to be between 0 and 1

        if is_sparse:
            rho_hat=tf.reduce_mean(normed(self.encode_weights)) #axis=0
            self.kl_loss=self.find_KL_div(self.rho,rho_hat)
            #self.kl_loss=tf.nn.l2_loss(self.weights1)
            self.cost=self.recon_loss+beta*self.kl_loss
        else:
            self.cost=self.recon_loss

        self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def find_KL_div(self,rho,rho_hat):
        return rho*tf.log(rho)-rho*tf.log(rho_hat)+(1-rho)*tf.log(1-rho)-(1-rho)*tf.log(1-rho_hat)

    def get_placeholders(self):
        inputs_=tf.placeholder(tf.float32,[None,self.in_dim])
        targets=tf.placeholder(tf.float32,[None,self.in_dim])
        lr=tf.placeholder(tf.float32)
        return inputs_,targets,lr

    def train(self,data,maxsize=-1,show_recon=False):
        """data: a Dataset object"""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            count=0
            for e in range(self.epochs):
                for batch_x,_ in data.train_batches(self.batch_size,sub_mean=True, maxsize=maxsize):
                    count+=1
                    feed={self.inputs_:batch_x,self.targets:batch_x,self.lr:self.learn_rate[e]}
                    _=sess.run([self.opt],feed_dict=feed)
                
                #print(f'Epoch {e+1}/{self.epochs}, recon_loss={rl}')

            all_weights=self.encode_weights.eval()

            #all_inputs=data.data.train.images[:maxsize]
            #all_inputs=all_inputs-all_inputs.mean(axis=1)[:,None]

            #feed={self.inputs_:all_inputs}
            #average_activations=sess.run(tf.reduce_mean(self.hlayer,axis=0),feed_dict=feed)
            
            #average_activations=average_activations[None,:]/average_activations.max()
            #print(average_activations)
            #all_weights=-np.abs(np.repeat(average_activations,784,axis=0)-np.maximum(0.,all_inputs).sum(axis=0)[:,None])
            if show_recon:
                test_x,_=data.test_set(maxsize=10,sub_mean=True)
                feed={self.inputs_:test_x}
                recons=sess.run(self.output,feed_dict=feed)
                return (all_weights,(test_x,recons))

        return all_weights

def plot_mAP(all_MAPs,hash_lengths,keys=['Fly','LSH']):
    k1,k2=keys
    mean= lambda x,n:np.mean(all_MAPs[x][n])
    stdev=lambda x,n:np.std(all_MAPs[x][n])
    colors={'LSH':'red','Fly':'green','SparseAEFly':'blue','DenseAEFly':'orange'}
    p=figure(x_range=[str(h) for h in hash_lengths],title=f'{k1} v/s {k2} on MNIST')
    delta=0.2 #,y_range=(0,0.5)
    x_axes=sorted([x-delta for x in range(1,1+len(hash_lengths))]+[x+delta for x in range(1,1+len(hash_lengths))])
    means=[mean(hashl,name) for name,hashl in zip([k1,k2]*len(hash_lengths),sorted(hash_lengths*2))]
    stds=[stdev(hashl,name) for name,hashl in zip([k1,k2]*len(hash_lengths),sorted(hash_lengths*2))]

    for i in range(len(hash_lengths)):
        p.vbar(x=x_axes[2*i], width=delta, bottom=0, top=means[2*i] , color=colors[k1],legend=k1)
        p.vbar(x=x_axes[2*i+1], width=delta, bottom=0, top=means[2*i+1] , color=colors[k2],legend=k2)

    err_xs=[[i,i] for i in x_axes]
    err_ys= [[m-s,m+s] for m,s in zip(means,stds)]
    p.y_range.bounds=(0,np.floor(10*max(means))/10 + 0.1)
    p.multi_line(err_xs, err_ys,line_width=2, color='black',legend='stdev')
    p.legend.location='top_left'
    p.legend.click_policy='hide'
    p.xaxis.axis_label='Hash length (k)'
    p.yaxis.axis_label='mean Average Precision (mAP)'
    output_file(k1+'_'+k2+'.html')
    show(p)


if __name__=='__main__':

    data=Dataset('mnist')
    input_dim=784 #d
    max_index=10000
    sampling_ratio=0.10
    nnn=200 #number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths=[2,4,8,12,16,20,24,28,32]
    inputs_=data.data.train.images[:max_index]
    all_MAPs={}
    for hash_length in hash_lengths: #k
        embedding_size= int(20*hash_length) #int(10*input_dim) #20k or 10d
        all_MAPs[hash_length]={}
        all_MAPs[hash_length]['Fly']=[]
        all_MAPs[hash_length]['LSH']=[]
        for _ in range(10):
            fly_model=flylsh(inputs_,hash_length,sampling_ratio,embedding_size)
            fly_mAP=fly_model.findmAP(nnn,1000)
            msg='mean average precision is equal to {:.2f}'.format(fly_mAP)
            #_=os.system('say "'+msg+'"') #works only on mac
            all_MAPs[hash_length]['Fly'].append(fly_mAP)
            
            dense_model=LSH(inputs_,hash_length)
            dense_mAP=dense_model.findmAP(nnn,1000)
            all_MAPs[hash_length]['LSH'].append(dense_mAP)
            msg='mean average precision is equal to {:.2f}'.format(dense_mAP)
            #_=os.system('say "'+msg+'"') #works only on mac
            print('Both models ran successfully')
        print(f'{hash_length} done')    
    
    print(all_MAPs)