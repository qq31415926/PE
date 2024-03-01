## source code for PE: A Poincare Explanation Method for Text Hierarchy Generation

For AOPC/Rotten Tomatoes/del/0.2, you can run like this:
`python test.py --dataset ptb --bert_path your_bert_path --class_num 2 --alpha1 0 --alpha2 0 --alpha3 1 --top 0.2 --del_or_pad del `

For AOPC/Yelp/del/0.2, you can run like this:
`python test.py --dataset ptb --bert_path your_bert_path --class_num 2 --alpha1 -0.5 --alpha2 -0.3 --alpha3 1 --top 0.2 --del_or_pad del `

For AOPC/TREC/del/0.2, you can run like this:
`python test.py --dataset trec --bert_path your_bert_path --class_num 6 --alpha1 0 --alpha2 0 --alpha3 1 --top 0.2 --del_or_pad del `

For time/Yelp, you can run like this:
`python visualize.py --dataset yelp --class_num 2 --alpha1 -0.5 --alpha2 -0.3 --alpha3 1`

