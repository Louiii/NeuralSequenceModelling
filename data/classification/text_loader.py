import torch

class TextLoader:
	def __init__(self, name='all_pytorch', rollout=25):
		data = data = open('data/classification/'+name+'.txt', 'r').read()[:10000]
		chars = sorted(list(set(data)))
		self.data_size, self.vocab_size = len(data), len(chars)
		print("----------------------------------------")
		print("Data has {} characters, {} unique".format(self.data_size, self.vocab_size))
		print("----------------------------------------")

		# char to index and index to char maps
		self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
		self.ix_to_char = { i:ch for i,ch in enumerate(chars) }

		# convert data from chars to indices
		data = list(data)
		for i, ch in enumerate(data):
		    data[i] = self.char_to_ix[ch]

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# data tensor on device
		data = torch.tensor(data).to(device)
		self.data = torch.unsqueeze(data, dim=1)

		self.rollout = rollout

	def __call__(self, t):
	    input_seq = self.data[t : t+self.rollout]
	    target_seq = self.data[t+1 : t+self.rollout+1]
	    return input_seq, target_seq