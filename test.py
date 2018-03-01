import lshutils
import numpy as np

if __name__=="__main__":
	data=lshutils.Dataset('random')
	max_index=10_000
	nnn=max_index//50
	inputs_=data.data[:max_index,:]

	lshmodel=lshutils.LSH(inputs_,hash_length=16)
	flymodel=lshutils.flylsh(inputs_,hash_length=16,sampling_ratio=0.1,embedding_size=20*16)

	lshmap=lshmodel.findmAP(nnn=nnn,n_points=100)
	flymap=flymodel.findmAP(nnn=nnn,n_points=100)

	print('LSH model mAP={:.3f}'.format(lshmap))
	print('Fly model mAP={:.3f}'.format(flymap))