# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:11:15 2016

@author: sonamsingh19
"""

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as LL
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import keras 
import lasagne

import data_utils

theano.config.floatX ='float32'

floatX=theano.config.floatX
dtype=floatX
dtensor5 = T.TensorType('float32', (False,)*5)

def init(shape):
    return keras.initializations.glorot_normal(shape)
    return (np.random.uniform(size=shape , low=-.01, high=.01).astype(floatX))

def init_bias(shape):
    
    return theano.shared(np.zeros(shape).astype(floatX), borrow=True)
def sample_weights(x,y):
    return init((x,y))


class Seq2Seq:
    def __init__(self ,input_vocab, output_vocab, buckets, hidden_sizes,num_layers,\
                                            max_gradient_norm,
                                            batch_size,
                                            learning_rate,
                                            lr_decay_factor,
                                            forward_only=False):
        pass
#    len(self.gr_vocab),
#                                            len(self.ph_vocab), self._BUCKETS,
#                                            size, num_layers,
#                                            max_gradient_norm,
#                                            batch_size,
#                                            learning_rate,False)
#                                            lr_decay_factor,
        self.buckets = buckets                           
#        input_vocab = {'_GO': 1, '_EOS': 2, u'L': 16, u"'": 4, '_PAD': 0, u'A': 5, u'C': 7, u'B': 6, u'E': 9, u'D': 8, u'G': 11, u'F': 10, u'I': 13, u'H': 12, u'K': 15, u'J': 14, u'M': 17, '_UNK': 3, u'O': 19, u'N': 18, u'Q': 21, u'P': 20, u'S': 23, u'R': 22, u'U': 25, u'T': 24, u'W': 27, u'V': 26, u'Y': 29, u'X': 28, u'Z': 30}
#        output_vocab = {u'IY': 21, '_GO': 1, u'W': 39, u'DH': 13, '_EOS': 2, u'Y': 40, u'HH': 19, u'CH': 11, u'JH': 22, u'ZH': 42, u'EH': 14, u'Z': 41, u'NG': 27, '_PAD': 0, u'TH': 35, u'AA': 4, u'B': 10, u'AE': 5, u'D': 12, u'G': 18, u'F': 17, u'AH': 6, u'K': 23, u'M': 25, '_UNK': 3, u'AO': 7, u'L': 24, u'IH': 20, u'S': 32, u'R': 31, u'EY': 16, u'T': 34, u'AW': 8, u'V': 38, u'AY': 9, u'N': 26, u'ER': 15, u'P': 30, u'UW': 37, u'SH': 33, u'UH': 36, u'OY': 29, u'OW': 28}
#        
        self.ignore_words=['_GO','_EOS','_PAD']
        
        self.batch_size =batch_size
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        
        self.rev_input_vocab = {v:k for k,v in self.input_vocab.iteritems()}
        self.rev_output_vocab = {v:k for k,v in self.output_vocab.iteritems()}

#        unique_vocab = list(set(self.input_vocab.values() + self.output_vocab.values()))
#        print(unique_vocab)
        
       
        values = list(self.input_vocab.items())
        values=sorted(values, key=lambda x:x[1])
        values=[i[1] for i in values]
        n_values = np.max(values) + 1

        self.onehot_input_vocab = np.eye(n_values)[values]
#          new_o= list(o.items())
#  
        values = list(self.output_vocab.items())
        values=sorted(values, key=lambda x:x[1])
        values=[i[1] for i in values]
        n_values = np.max(values) + 1

        self.onehot_output_vocab = np.eye(n_values)[values]
        
#        print(onehot_vocab)
        self.input_width = len(input_vocab)
        
        self.input_var, self.target_var = T.ftensor3s('x','y')
        self.input_vocab_size = len(input_vocab)
        self.output_vocab_size = len(output_vocab)
        self.output_init =theano.shared( np.vstack([self.onehot_output_vocab[1] \
        for j in range(self.batch_size)]).astype(np.float32))
        
        print('vocab size', self.input_vocab_size, self.output_vocab_size)
    
        input_size= self.input_vocab_size 
        embed_size= 400
        n_in = embed_size# for embedded reber grammar
        n_hidden = n_i = n_c = n_o = n_f =512
        n_y = self.output_vocab_size # for embedded reber grammar
        
        # initialize weights
        # i_t and o_t should be "open" or "closed"
        # f_t should be "open" (don't forget at the beginning of training)
        # we try to archive this by appropriate initialization of the corresponding biases 
        self.enc_params=[]
        for i in range(2):
            W_xi =sample_weights(n_in, n_i)
            W_hi = sample_weights(n_hidden, n_i)
            W_ci = sample_weights(n_c, n_i)
            b_i = init_bias((n_i,))
            W_xf =sample_weights(n_in, n_f)
            W_hf = sample_weights(n_hidden, n_f)
            W_cf =sample_weights(n_c, n_f)
            b_f =  init_bias((n_f,))
            W_xc = sample_weights(n_in, n_c)
            W_hc = sample_weights(n_hidden, n_c)
            b_c =  init_bias((n_c,) )
            W_xo = sample_weights(n_in, n_o)
            W_ho = sample_weights(n_hidden, n_o)
            W_co = sample_weights(n_c, n_o)
            b_o = init_bias((n_o,))
            p=[W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo,\
            W_ho, W_co, b_o]
            self.enc_params.append(p)
            print(len(p))
#            n_in=n_hidden

        
        n_in=n_y
        self.dec_params=[]
#  
        for i in range(2):
            W_xi =sample_weights(n_in, n_i)
            W_hi = sample_weights(n_hidden, n_i)
            W_ci = sample_weights(n_c, n_i)
            b_i = init_bias((n_i,))
            W_xf =sample_weights(n_in, n_f)
            W_hf = sample_weights(n_hidden, n_f)
            W_cf =sample_weights(n_c, n_f)
            b_f =  init_bias((n_f,))
            W_xc = sample_weights(n_in, n_c)
            W_hc = sample_weights(n_hidden, n_c)
            b_c =  init_bias((n_c,) )
            W_xo = sample_weights(n_in, n_o)
            W_ho = sample_weights(n_hidden, n_o)
            W_co = sample_weights(n_c, n_o)
            b_o = init_bias((n_o,))
            p=[W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo,\
            W_ho, W_co, b_o]
            self.dec_params.append(p)
            print(len(p))
        self.W_hy = sample_weights(n_hidden, n_y)
        self.b_y = init_bias((n_y,) )
        self.embed =sample_weights(input_size, embed_size)
#        self.embed_dec =sample_weights(self.output_vocab_size, embed_size)

        self.dim_proj=n_hidden  
        self.c0 = theano.shared(np.zeros((self.batch_size,n_hidden), dtype=dtype))
        self.h0 = theano.shared(np.zeros((self.batch_size,n_hidden), dtype=dtype))
        
        self.c0_1 = theano.shared(np.zeros((self.batch_size,n_hidden), dtype=dtype))
        self.h0_1 = theano.shared(np.zeros((self.batch_size,n_hidden), dtype=dtype))
        print('hidden sies',self.c0.get_value().shape)
        print('hidden sies',self.h0.get_value().shape)
        print('hidden sies',self.output_init.get_value().shape)
        
    

        self.total_params= self.enc_params[0]+self.enc_params[1]+\
        self.dec_params[0]+ self.dec_params[1]+\
        [ self.W_hy ,self.b_y, self.embed]

        self.all_params =self.total_params
        self.train_fn, self.compute_cost_fn, self.probs_fn= self.build_net()
        
    def _slice(self,state,ind, dim_proj):
#        assert state.ndim==3
        return state[:,ind*dim_proj :(ind+1)*dim_proj]
#        

   
    
    
    def _step2(self,x_t, h_tm1, c_tm1, x_w, h_w, c_w ,W_co, b_i, b_f,  b_c,  b_o):

        sigma = lasagne.nonlinearities.sigmoid

        
        # for the other activation function we use the tanh
        act = T.tanh
     
        # sequences: x_t
        # prior results: h_tm1, c_tm1
        # non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_cy, b_y
        x_prod = theano.dot(x_t,x_w)
        h_prod = theano.dot(h_tm1,h_w)
        c_prod = theano.dot(c_tm1,c_w)


        i_t = sigma(self._slice(x_prod,0,self.dim_proj) + self._slice(h_prod,0,self.dim_proj)  + \
        self._slice(c_prod,0,self.dim_proj)  + b_i.dimshuffle(('x',0)))
        
        f_t = sigma(self._slice(x_prod,1,self.dim_proj) + self._slice(h_prod,1,self.dim_proj)  + \
        self._slice(c_prod,1,self.dim_proj)  \
        + b_f.dimshuffle(('x',0)))
        
        c_t = f_t * c_tm1 + i_t * act(self._slice(x_prod,2,self.dim_proj) + self._slice(h_prod,2,self.dim_proj)   \
        + b_c.dimshuffle(('x',0)) )
        
        o_t = sigma(self._slice(x_prod,3,self.dim_proj)+ self._slice(h_prod,3,self.dim_proj) +\
          theano.dot(c_t, W_co) + b_o.dimshuffle(('x',0)))
          
        h_t = o_t * act(c_t)

        return [h_t, c_t]
        
    def _dec_step2(self,x_t,  h_tm1, c_tm1,h_tm2, c_tm2, y_t, x_w, h_w, c_w ,W_co, b_i, b_f,  b_c,  b_o,\
     x_w1, h_w1, c_w1 ,W_co1, b_i1, b_f1,  b_c1,  b_o1, W_hy,b_y):
        
#        W_ctxi, W_xi, W_hi, W_ci, b_i, W_ctxf,W_xf, W_hf, W_cf, b_f, W_ctxc,W_xc, W_hc, b_c, W_ctxo,W_xo, W_ho, W_co, b_o =self.dec_params[0]
        
        sigma = lasagne.nonlinearities.sigmoid

        x_t =y_t
#        ctx= self.c_tx[0]
        # for the other activation function we use the tanh
        act = T.tanh
 
        x_prod = theano.dot(x_t,x_w)
        h_prod = theano.dot(h_tm1,h_w)
        c_prod = theano.dot(c_tm1,c_w)

        # sequences: x_t
        # prior results: h_tm1, c_tm1
        # non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_cy, b_y
       
        i_t = sigma( self._slice(x_prod,0,self.dim_proj) + self._slice(h_prod,0,self.dim_proj)  + \
        self._slice(c_prod,0,self.dim_proj)  + b_i.dimshuffle(('x',0)))
        
        f_t = sigma(( self._slice(x_prod,1,self.dim_proj) + self._slice(h_prod,1,self.dim_proj)  + \
        self._slice(c_prod,1,self.dim_proj)  \
        + b_f.dimshuffle(('x',0))))
        
        c_t1 = f_t * c_tm1 + i_t * act(self._slice(x_prod,2,self.dim_proj) + self._slice(h_prod,2,self.dim_proj)   \
        + b_c.dimshuffle(('x',0)) )
        
        o_t1 = sigma(self._slice(x_prod,3,self.dim_proj)+ self._slice(h_prod,3,self.dim_proj) +\
          theano.dot(c_t1, W_co) + b_o.dimshuffle(('x',0)))
          
        h_t1 = o_t1 * act(c_t1)
#        y_t = T.nnet.softmax((theano.dot(h_t, self.W_hy) + self.b_y.dimshuffle(('x',0))))
       
       
      
        x_t =y_t
        
        
        h_tm1 = h_tm2
        c_tm1=c_tm2
     
        x_prod = theano.dot(x_t,x_w1)
        h_prod = theano.dot(h_tm1,h_w1)
        c_prod = theano.dot(c_tm1,c_w1)

        # sequences: x_t
        # prior results: h_tm1, c_tm1
        # non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_cy, b_y
       
        i_t = sigma(self._slice(x_prod,0,self.dim_proj) + self._slice(h_prod,0,self.dim_proj)  + \
        self._slice(c_prod,0,self.dim_proj)  + b_i1.dimshuffle(('x',0)))
        
        f_t = sigma( self._slice(x_prod,1,self.dim_proj) + self._slice(h_prod,1,self.dim_proj)  + \
        self._slice(c_prod,1,self.dim_proj)  \
        + b_f1.dimshuffle(('x',0)))
        
        c_t2 = f_t * c_tm1 + i_t * act(self._slice(x_prod,2,self.dim_proj) + self._slice(h_prod,2,self.dim_proj)   \
        + b_c1.dimshuffle(('x',0)) )
        
        o_t2 = sigma(self._slice(x_prod,3,self.dim_proj)+ self._slice(h_prod,3,self.dim_proj) +\
          theano.dot(c_t2, W_co1) + b_o1.dimshuffle(('x',0)))
          
        h_t2 = o_t2 * act(c_t2)
        
        h_t2= T.add(h_t1, h_t2)
        y_t = T.nnet.softmax((theano.dot(h_t2, W_hy) + b_y.dimshuffle(('x',0))))
        return [h_t1, c_t1,h_t2, c_t2, y_t]
            
            
        

    def  enc_dec(self):

         W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o=\
         self.enc_params[0]
         x_w = T.concatenate((W_xi, W_xf, W_xc, W_xo),axis=1)
         h_w = T.concatenate((W_hi, W_hf, W_hc, W_ho),axis=1)
         c_w = T.concatenate((W_ci, W_cf),axis=1)

         non_sec =[x_w, h_w, c_w ,W_co, b_i, b_f,  b_c,  b_o]
         
         embeddings,_ =theano.scan (lambda x : (theano.dot(x,self.embed)), sequences =self.input_var.dimshuffle((1,0,2)))
        
         hids1, updates =theano.scan(fn =self._step2, sequences = embeddings,\
         outputs_info=[self.h0,self.c0], non_sequences = non_sec, strict=True)
                    
        
         W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o=\
         self.enc_params[1]
         x_w = T.concatenate((W_xi, W_xf, W_xc, W_xo),axis=1)
         h_w = T.concatenate((W_hi, W_hf, W_hc, W_ho),axis=1)
         c_w = T.concatenate((W_ci, W_cf),axis=1)
         non_sec =[x_w, h_w, c_w ,W_co, b_i, b_f,  b_c,  b_o]

         hids2, updates =theano.scan(fn =self._step2, sequences = embeddings,\
         outputs_info=[self.h0_1,self.c0_1], non_sequences = non_sec,go_backwards=True, strict=True)
         
         
         bi_lstm_out =T.add(hids1[0], hids2[0]) 
         out_init =T.nnet.softmax((theano.dot(bi_lstm_out[-1], self.W_hy) + self.b_y.dimshuffle(('x',0))))

         embeddings_dec,_ =theano.scan (lambda x : x, sequences =self.target_var.dimshuffle((1,0,2)))
         
       
         W_xi, W_hi, W_ci, b_i,W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, \
          W_xo, W_ho, W_co, b_o= self.dec_params[0]
            

         x_w = T.concatenate((W_xi, W_xf, W_xc, W_xo),axis=1)
         h_w = T.concatenate((W_hi, W_hf, W_hc, W_ho),axis=1)
         c_w = T.concatenate((W_ci, W_cf),axis=1)
         
         W_xi, W_hi, W_ci, b_i1,W_xf, W_hf, W_cf, b_f1, W_xc, W_hc, b_c1, \
        W_xo, W_ho, W_co1, b_o1= self.dec_params[1]
            

         x_w1 = T.concatenate((W_xi, W_xf, W_xc, W_xo),axis=1)
         h_w1 = T.concatenate((W_hi, W_hf, W_hc, W_ho),axis=1)
         c_w1 = T.concatenate((W_ci, W_cf),axis=1)
         
         non_sec_dec= [ x_w, h_w, c_w ,W_co, b_i, b_f,  b_c,  b_o,\
          x_w1, h_w1, c_w1 ,W_co1, b_i1, b_f1,  b_c1,  b_o1, self.W_hy,self.b_y,
         ]
         
         outs, updates =theano.scan(fn =self._dec_step2, sequences =embeddings_dec,\
         outputs_info=[hids1[0][-1],hids1[1][-1],hids2[0][-1],hids2[1][-1],self.output_init],\
         non_sequences= non_sec_dec, strict=True)
#        
         return outs[4].dimshuffle((1,0,2))
        
    def build_net(self):
        
    
        # Input layer
       
        # reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing 
        #us to use different batch sizes in the model.
        output =self.enc_dec()
        # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
        cost = T.nnet.categorical_crossentropy( output.reshape((-1,self.output_vocab_size)),\
        self.target_var.reshape((-1,self.output_vocab_size))).mean()
    
        # Retrieve all parameters from the network
#        all_params = lasagne.layers.get_all_params(self.1,trainable=True)
    
        # Compute AdaGrad updates for training
        print("Computing updates ...")
        print('params')
        for p in self.all_params:
            print(p.get_value ().shape)
            
        lr =T.scalar('lr')
        all_grads =T.grad(cost,self.all_params)
#        all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

        updates = lasagne.updates.rmsprop(all_grads, self.all_params,\
                                          learning_rate=lr)

#        updates = lasagne.updates.rmsprop(cost, self.all_params,lr)
    
        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function( [self.input_var, self.target_var,lr], cost, updates=updates, allow_input_downcast=True)
        compute_cost = theano.function([self.input_var, self.target_var], cost, allow_input_downcast=True)
    
        # In order to generate text from the network, we need the probability distribution of the next character given
        # the state of the network and the input (a seed).
        # In order to produce the probability distribution of the prediction, we compile a function called probs. 
        
        probs = theano.function([self.input_var,  self.target_var],output,allow_input_downcast=True)
        
        return train, compute_cost, probs
        
 
        
    def pad(self, bucket, source, target): 
#        print(source, target)
#        print(self.rev_input_vocab, self.rev_output_vocab)
        pad_symbol = [self.input_vocab['_PAD']]
#        target =  + target + [self.output_vocab['_EOS']]
#        print(bucket, source)
        pad_source =abs(len(source)-bucket[0])
        
        
        pad_target =  abs(len(target)-bucket[1])
#        print(pad_source, pad_target, bucket)
        assert len( pad_source * pad_symbol + source)==bucket[0]
        assert  len(target +  pad_target* pad_symbol)==bucket[1]
        
        return ( pad_source * pad_symbol + source, target +  pad_target* pad_symbol)
#        print (pad_source, pad_target)
        
    def train(self, inputs):
        
        x_1hots, y_1hots=[],[]
#        print(inputs)
        lr = inputs[-1]
        inputs= inputs[:-1]
        for example in inputs:
            x,y =example
#            print(x, 'transformed', self.onehot_vocab[x])
#            break
            x_1hot, y_1hot =self.onehot_input_vocab[x], self.onehot_output_vocab[y]
#            print(x_1hot.shape, y_1hot.shape)
#            print(x,y)
            x_1hots.append(x_1hot[None])
            y_1hots.append(y_1hot[None])
#            print(x_1hot[None].shape)
#            print(y_1hot[None].shape)
#                              
        x_1hots =np.vstack(x_1hots)
        y_1hots = np.vstack(y_1hots)
#        print('x',np.argmax(x_1hots,axis=2))
#        print('y',np.argmax(y_1hots,axis=2))

#        print('train,shape', x_1hots.shape, y_1,lrhots.shape)
        tr_cost = self.train_fn(x_1hots, y_1hots,lr)
#            print(' log loss',tr_cost)
            
#        print(tr_cost.shape)
        return tr_cost# sum(loss)/len(loss)
    def decode_all (self, inputs):
        '''
        expects in batch
        '''
      
        x_1hots, y_1hots=[],[]
#        print('l',inputs)
        for example in  inputs:
#            print(example,'d')
            word ,target =example
            source_ids, target_ids = word, target
#            if len(target)>1:
#                print(len(target),target)
#            print(source_ids, target_ids)
#            s_ids=[]
#            t_ids=[]
            flag = False
            for gr in source_ids:
                if gr not in self.rev_input_vocab.keys():
                    flag =True
                    print gr, flag

            for tr in target_ids:
                if tr not in self.rev_output_vocab.keys():
                    flag =True
                    print tr, flag
            if flag:
                continue
##            print(target)
#            for gr in source_ids:
#                s_ids += [self.input_vocab[gr]]
#        
#            for tr in target_ids:
#                t_ids += [self.output_vocab[tr]]
#            
#            target_ids.append(data_utils.EOS_ID)

#            for bucket_id, (source_size, target_size) in enumerate(self.buckets):
#                if len(source_ids) < source_size and len(target_ids) < target_size:
#                    s , t =self.pad(self.buckets[bucket_id],\
#                  source_ids, target_ids)
##                    print('s',s,source_ids,'t',t,target_ids)
#                    break
#            print('source_ids',source_ids)
#            print('target_ids',target_ids)
            x_1hots.append(self.onehot_input_vocab[source_ids][None])
            y_1hots.append(self.onehot_output_vocab[target_ids][None])
  
        x_1hots =np.vstack(x_1hots)
        y_1hots = np.vstack(y_1hots)
#        print('test x',np.argmax(x_1hots,axis=2))
#        print('test y',np.argmax(y_1hots,axis=2))
##        
        probs = self.probs_fn(x_1hots,y_1hots )
#        print(probs.shape)
        preds =np.argmax(probs , axis=2)
#        print ('input',inputs)
#        print('target',)
#        print('x1hots',np.argmax(x_1hots,axis=2)[0])
#        
#        print('xyhots',np.argmax(y_1hots,axis=2)[0])

        final_preds =[]
        originals =[]
        words_input =[]
        for i, pred_symbols in enumerate(preds):
            
            produced_ph =[]

            for symbol in pred_symbols:
#                print(symbol)
#                if symbol not in self.ignore_words:
##                    print ('inside ignorew words' )
                    produced_ph.append(self.rev_output_vocab[symbol])
            final_preds.append(' '.join(produced_ph))
            
            originals.append([' '.join(self.rev_output_vocab[inp] for inp in inputs[i][1])])
            words_input.append([' '.join(self.rev_input_vocab[inp] for inp in inputs[i][0])])

#        print('produced',  ' '.join(produced_ph))
            
        return final_preds,  words_input, originals
