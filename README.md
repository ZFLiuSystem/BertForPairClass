>**Overview**
> -
> <div style="text-align: justify;">This engineering project revolves around a task in the field of NLP - textual entailment. 
> At the heart of the project lies the adoption of the pre-trained language model BERT 
> (Bidirectional Encoder Representations from Transformers) provided by HuggingFace as the foundational framework. 
> The model underwent supervised fine-tuning, involving a series of adjustments and optimizations. Through these efforts, 
> the model has successfully captured logical relationships and entailment information between texts, 
> demonstrating accuracy in textual entailment task. Currently, all datasets, model, and files and codes related to 
> training and testing used in this project have been open-sourced for public access and use.</div>

> **BERT**  
> -
> _Reference_: https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional.
><div style="text-align: justify;">BERT (Bidirectional Encoder Representations from Transformers), 
> which was introduced by Google in 2018, 
> is fundamentally rooted in the concept of bidirectional self-attention. 
> It operates as an Encoder-only Transformer model, 
> focusing solely on the encoding phase without a corresponding decoder component. 
> This architectural choice allows BERT to deeply understand and process textual data by considering both the left 
> and right contexts simultaneously, thereby capturing a more comprehensive and nuanced understanding of language. 
> By leveraging this powerful mechanism, BERT has proven to be highly effective in a wide range of NLP tasks, 
> including but not limited to textual entailment, where it has demonstrated remarkable accuracy and performance.</div>

> **Environment**
> -
> * **Hardware:**<br>CPU == AMD Ryzen 7 5800H;<br>GPU == NVIDIA GeForce RTX3070 Laptop (8GB);
> <br>RAM == Micron Technology DDR4 (16GB).
> ---
> * **Software:** <br>IDE == Pycharm 2023.2;<br> Python == 3.11;<br> Pytorch == 2.2.1;
> <br> transformers == 4.42.3.
> ---
> * **Notice:**<br>
> There are some compatibility issues between the versions of Pytorch and transformers, 
> so some warnings will be reported by Pycharm during training and test.


> **Training&Test**
> -
> * **Model:** <br>
> BertForPairClass(<br>
  (bert): BertModel(<br>
    (embeddings): BertEmbeddings(<br>
      (word_embeddings): Embedding(30522, 768, padding_idx=0)<br>
      (position_embeddings): Embedding(512, 768)<br>
      (token_type_embeddings): Embedding(2, 768)<br>
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<br>
      (dropout): Dropout(p=0.1, inplace=False)
    )<br>
    (encoder): BertEncoder(<br>
      (layer): ModuleList(<br>
        (0-11): 12 x BertLayer(<br>
          (attention): BertAttention(<br>
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)<br>
              (key): Linear(in_features=768, out_features=768, bias=True)<br>
              (value): Linear(in_features=768, out_features=768, bias=True)<br>
              (dropout): Dropout(p=0.1, inplace=False)
            )<br>
            (output): BertSelfOutput(<br>
              (dense): Linear(in_features=768, out_features=768, bias=True)<br>
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<br>
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )<br>
          (intermediate): BertIntermediate(<br>
            (dense): Linear(in_features=768, out_features=3072, bias=True)<br>
            (intermediate_act_fn): GELUActivation())<br>
          (output): BertOutput(<br>
            (dense): Linear(in_features=3072, out_features=768, bias=True)<br>
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<br>
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )<br>
    (pooler): BertPooler(<br>
      (dense): Linear(in_features=768, out_features=768, bias=True)<br>
      (activation): Tanh()
    )
  )<br>
  (dropout): Dropout(p=0.3, inplace=False)<br>
  (classifier_head): Linear(in_features=768, out_features=3, bias=True)<br>
  (loss_fn): CrossEntropyLoss()
)<br>
> ---
> * **TrainingPart:** <br>
> ![Training_](Graph_/training_.png)
> ---
> * **TestPart:** <br>
> ![Test_](Graph_/test_.png)