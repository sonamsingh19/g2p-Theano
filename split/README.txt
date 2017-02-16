Standard CMUDict g014b2b training/testing run

Performance on this test is reported in various g2p papers, in
particular Phonetisaurus IS2013 paper (Failure transitions for Joint
n-gram Models and G2P Conversion Josef R. Novak, Nobuaki Minematu,
Keikichi Hirose)

It is also referred as g014b2b set.

Phonetisaurus baseline is %24.4 as in is2014-conversion archive

```
phonetisaurus-align --input=g014b2b.train --ofile=g014b2b/g014b2b.corpus --seq1_del=false

train-ngramlibrary.py --prefix "g014b2b/g014b2b" --order 8 --bins 3
                        
phonetisaurus-calculateER-omega --testfile g014b2b.test --modelfile g014b2b/g014b2b8.fst \
        --prefix "g014b2b/g014b2b8" --decoder_type fst_phi
```

g2p-seq2seq result is %24.4

```
python /home/ubuntu/g2p-seq2seq/g2p_seq2seq/g2p.py --train \
    cmudict.dic.train --test cmudict.dic.test --num_layers 2 --size 512 '
    --model model --max_steps 0
```
