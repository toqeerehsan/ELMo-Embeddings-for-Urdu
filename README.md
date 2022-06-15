# ELMo Embeddings for Urdu

ELMo embeddings are introduced by [AllenNLP](https://allenai.org/allennlp/software/elmo). ELMo is a deep contextualized word representation that models linguistic contexts. The word vectors are trained on an Urdu text corpus containing 220 millions words. 

**Repository**
```
urdu_vocabulary.txt # Vocabulary size is 125,622
options.json
weights.hdf5
```
**Usage**
```
print("Loading pre-trained ELMo embeddings...")
sent_length = 100
elmo_dim = 128  # trained for 128 dimensions
elmo_layer = 2  # layers are 0,1,2

sentence = "".strip().split()
from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder(options_file='urdu_elmo_220m/options.json',weight_file='urdu_elmo_220m/weights.hdf5')
vectors = elmo.embed_sentence(sentence)
elmo_vec = np.zeros( (sent_length, elmo_dim) )
elmo_vec[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
elmo_vec = np.expand_dims(elmo_test_x, axis=0)  
# elmo_vec contains the vectors for each token in the sentence
```
**Citation**

If you use Urdu ELMo embeddings in your research then please cite the following paper:
```
@article{ehsan2021improving,
  title={Improving Phrase Chunking by using Contextualized Word Embeddings for a Morphologically Rich Language},
  author={Ehsan, Toqeer and Khalid, Javairia and Ambreen, Saadia and Mustafa, Asad and Hussain, Sarmad},
  journal={Arabian Journal for Science and Engineering},
  pages={1--19},
  year={2021},
  publisher={Springer}
}
```
