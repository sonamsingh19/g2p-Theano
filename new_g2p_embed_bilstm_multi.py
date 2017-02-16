# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:27:28 2016

@author: sonamsingh19
"""
import data_utils
import math
import numpy as np
import time
from Seq_Model_LSTM_peek_concat_bilstm_multi import Seq2Seq
import theano
from sklearn.cross_validation import train_test_split
import pandas as pd
class G:

    _BUCKETS = [(5, 10), (10, 15),(15,20),(20,25),(25,30), (40, 50)]
    
    def __init__(self):
        
        print("Preparing G2P data")
        train_path = 'split/cmudict.dic.train'
        test_path ='split/cmudict.dic.test'
        valid_path ='split/cmudict.dic.test'
        self.model_dir=''
        num_layers = 1
        size =10
        max_gradient_norm= 3
        self.batch_size  =32
        self.learning_rate = 1e-3
        lr_decay_factor=0        
        
        self.train_gr_ids, self.train_ph_ids, self.valid_gr_ids, self.valid_ph_ids, self.gr_vocab,\
        self.ph_vocab, self.test_lines =  data_utils.prepare_g2p_data(self.model_dir, train_path, valid_path,
                                    test_path)
        
#        pd.DataFrame(self.valid_gr_ids).to_csv('ids')
        print(self.gr_vocab)
        
        print(self.ph_vocab)
#        test_dic = data_utils.collect_pronunciations(self.test_lines)
#        
#        count=0
#        for i in (test_dic.values()):
#            print len(i)
#            if len(i)>1:
#                count+=1
#        print(count)

        train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids=\
                  self.train_gr_ids, self.train_ph_ids, self.valid_gr_ids, self.valid_ph_ids                    
        print ("Reading development and training data.")
      
    
        self.model =Seq2Seq(self.gr_vocab,
                                            self.ph_vocab, self._BUCKETS,
                                            size, num_layers,
                                            max_gradient_norm,
                                            self.batch_size,
                                            self.learning_rate,
                                            lr_decay_factor,
        
                                          False)
                                         
        self.test_dic = data_utils.collect_pronunciations(self.test_lines)

        m =0
        for i in self.valid_gr_ids:
            if len(i)>m:
                m=len(i)
        print('max range valid gr', m)
        m =0
        for i in self.valid_ph_ids:
            if len(i)>m:
                m=len(i)
        print('max range valid ph', m)
        
        m =0
        for i in self.train_gr_ids:
            if len(i)>m:
                m=len(i)
        print('max range train gr', m)
        m =0
        for i in self.train_ph_ids:
            if len(i)>m:
                m=len(i)
        print('max range train ph', m)
        self.valid_set = self.__put_into_buckets(valid_gr_ids, valid_ph_ids)
        self.train_set = self.__put_into_buckets(train_gr_ids, train_ph_ids)
        
        self.train_set =self.train_set[:-2]
        self.valid_set =self.valid_set[:-2]
        for i,t in enumerate(self.train_set):
            print('train_set sizes', len(t), len(train_gr_ids) )
#            pd.DataFrame(t).to_csv(str(i)+'.csv')
        self.rev_ph_vocab = dict([(x, y) for (y, x) in enumerate(self.ph_vocab)])
        print("Creating %d layers of %d units." % (num_layers, size))
    
        self.train()
 
      
   
    def train(self):
       
            # This is the training loop.
            max_steps= 1000
  

            prev_min_loss =10
            lr= self.learning_rate
            self.evaluate_batch()
            for i in range(max_steps):
                losses=[]
                perps=[]
                for X_train in self.train_set:
                   start_time = time.time()
#                   print('sample',X_train[-2:])

                   print('training size m training on',len(X_train))
                   for batch in range(1, len(X_train), self.batch_size):
                          if batch +self.batch_size> len(X_train):
                             print('last batch')
                             inputs = X_train[-self.batch_size:]
                          else:
                             inputs = X_train[batch:batch + self.batch_size]
        #                  loss =
                          # Get a batch and make a step.
    #                      print('train,shape')
                          loss = self.model.train(inputs+[lr]) 
    
                          perplexity = math.exp(loss) if loss < 300 else float('inf')
                       
                          losses+=[loss]
                          perps+= [perplexity]
                   print('time for set', time.time()-start_time)
                print('epoch:',i)              
                print('loss',np.array(losses).mean())
                print ('perp',np.array(perps).mean())
                if np.around(np.array(losses).mean(),3)< np.around(prev_min_loss,3):
                    print('prev min loss',prev_min_loss, ' current loss',np.array(losses).mean())
                    prev_min_loss =np.around(np.array(losses).mean(),3)
                    print('learning rate halved to',lr)

                else:
#                    lr=lr/2.0
                    print('learning rate halved to',lr)
                self.evaluate_batch()

            print('Training done.')


    def return_par(self):
        return   (self.train_gr_ids, self.train_ph_ids, self.valid_gr_ids, self.valid_ph_ids, self.gr_vocab,\
    self.ph_vocab, self.test_lines)
 
    def __put_into_buckets(self, source, target):
        """Put data from source and target into buckets.
    
        Args:
          source: data with ids for graphemes;
          target: data with ids for phonemes;
            it must be aligned with the source data: n-th line contains the desired
            output for n-th line from the source.
    
        Returns:
          data_set: a list of length len(_BUCKETS); data_set[n] contains a list of
            (source, target) pairs read from the provided data that fit
            into the n-th bucket, i.e., such that len(source) < _BUCKETS[n][0] and
            len(target) < _BUCKETS[n][1]; source and target are lists of ids.
        """
    
        # By default unk to unk
        data_set = [[[[4],[4]]] for _ in self._BUCKETS]
    
        for source_ids, target_ids in zip(source, target):
          
          target_ids.append(data_utils.EOS_ID)
#          target_ids =[data_utils.GO_ID]+target_ids
          for bucket_id, (source_size, target_size) in enumerate(self._BUCKETS):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append( self.model.pad(self._BUCKETS[bucket_id],\
              source_ids, target_ids))
              break
    
        return data_set
#        
#    def evaluate(self, test_lines):
#        """Calculate and print out word error rate (WER) and Accuracy
#           on test sample.
#    
#        Args:
#          test_lines: List of test dictionary. Each element of list must be String
#                    containing word and its pronounciation (e.g., "word W ER D");
#        """
#        if not hasattr(self, "model"):
#          raise RuntimeError("Model not found in %s" % self.model_dir)
#    
#        test_dic = data_utils.collect_pronunciations(test_lines)
#        
#        
#        if len(test_dic) < 1:
#            print("Test dictionary is empty")
#            return
#    
#        print('Beginning calculation word error rate (WER) on test sample.')
#        errors = self.calc_error(test_dic)
#    
#        print("Words: %d" % len(test_dic))
#        print("Errors: %d" % errors)
#        print("WER: %.3f" % (float(errors)/len(test_dic)))
#        print("Accuracy: %.3f" % float(1-(errors/len(test_dic))))
    def evaluate_batch(self,):
        """Calculate and print out word error rate (WER) and Accuracy
           on test sample.
    
        Args:
          test_lines: List of test dictionary. Each element of list must be String
                    containing word and its pronounciation (e.g., "word W ER D");
        """
        if not hasattr(self, "model"):
          raise RuntimeError("Model not found in %s" % self.model_dir)
        print('Beginning calculation word error rate (WER) on test sample.')
        errors,acc = self.calc_error_batch()
        l =0 
        for X_test in self.valid_set:
           
           for batch in range(1, len(X_test), self.batch_size):
                  if batch +self.batch_size> len(X_test):
                    
                     inputs = X_test[-self.batch_size:]
                  else:
                     inputs = X_test[batch:batch + self.batch_size] 
                  l+= len(inputs)
            
        print("Words: %d" %  l)
        print("Errors: %d" % errors)
        print("correct: %d" % acc)

        print("WER: %.3f" % (float(errors)/l))
        print("Accuracy: %.3f" % float(1-(errors/l)))
        
    def strip_pads(self,w)    :
        new_w = []
        for letter in w:
#            print (letter)
            if letter not in self.model.ignore_words:
                new_w.append(letter)
        return new_w
        
    def calc_error_batch(self):
        """Calculate a number of prediction errors.
        """
        errors = 0
        acc=0
        W=[]
        #        test_dic = data_utils.collect_pronunciations(test_lines)

        for X_test in self.valid_set:
           print('testing size on',len(X_test))
           for batch in range(1, len(X_test), self.batch_size):
                  if batch +self.batch_size> len(X_test):
                     print('last batch')
                     inputs = X_test[-self.batch_size:]
                  else:
                     inputs = X_test[batch:batch + self.batch_size] 
#                  print('fed to decode_all', inputs)
                  outs, words, pronunciations = self.model.decode_all(inputs)
#                         if len(pronunciations) >1: #u'CHALETS': [u'SH AE L EY Z', u'SH AH L EY Z']
                  
                  for hyp,word, pronunciation in  zip(outs,words,pronunciations):
                      hyp=' '.join( self.strip_pads(hyp.split(' ' )))
#                      pronunciation = strip_pads(pronunciation)
                      word= self.strip_pads(word[0].split(' '))
                      pronunciation = self.test_dic[''.join(word)]
                      W.append(('first',word,pronunciation,hyp, len(hyp) ,len(pronunciation)))
                      if hyp not in pronunciation:
                            errors += 1
                      else:
                            acc +=1
                
        print(W[:5])
        return errors, acc
#    def calc_error(self, dictionary):
#        """Calculate a number of prediction errors.
#        """
#        errors = 0
#        W=[]
#        sources, targets=[] , []
#        for word, pronunciations in dictionary.items():
#            if len(pronunciations) >1: #u'CHALETS': [u'SH AE L EY Z', u'SH AH L EY Z']
##                targets.append(pronunciations[0])
##            else:
##                 targets.append(pronunciations)
##
##            sources.append(word)
##        for source, target in zip(sources,targets):
##                print([word]*self.batch_size,[pronunciations[0]]*self.batch_size,'first',pronunciations)
#                hyp = self.model.decode([word]*self.batch_size,[pronunciations[0]]*self.batch_size)
#                W.append(('first',word,pronunciations,hyp[0],hyp[0], len(hyp) ,len(pronunciations)))
#            else: #u'FRANCIES': [u'F R AH N S IY Z']
##                print([word]*self.batch_size,\
##                [pronunciations] *self.batch_size,'secoind',pronunciations)
#                hyp = self.model.decode([word]*self.batch_size,\
#                pronunciations *self.batch_size)
#                W.append(('second',word,pronunciations,hyp[0],hyp[0],len(hyp),len(pronunciations)))
#            if hyp[0] not in pronunciations:
#                errors += 1
#            
#        print(W[:5])
#        return errors
        
  
x= G()