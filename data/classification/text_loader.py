import torch

class TextLoader:
	def __init__(self, name='all_pytorch', rollout=25, maxlen=10000):
		data = open('data/classification/'+name+'.txt', 'r').read()[:maxlen]
		chars = sorted(list(set(data)))
		self.data_size, self.vocab_size = len(data), len(chars)
		print(str(self.data_size)+' characters, '+str(self.vocab_size)+' unique')

		# char to index and index to char
		self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
		self.ix_to_char = { i:ch for i,ch in enumerate(chars) }

		# convert data, chars to indices
		data = [self.char_to_ix[ch] for ch in list(data)]

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.data = torch.unsqueeze(torch.tensor(data).to(device), dim=1)

		self.rollout = rollout

	def __call__(self, t):
	    input_seq = self.data[t : t+self.rollout]
	    target_seq = self.data[t+1 : t+self.rollout+1]
	    return input_seq, target_seq