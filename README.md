# Robustness of DistilBART-cnn-12-6 in Dialogue Summarization

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Citation](#citation)
- [Licensing Information](#licensing-information)
- [Contributions](#contributions)
- [Acknowledgements](#acknowledgements)

## Introduction

This project explores the robustness of the DistilBART-cnn-12-6 model in text summarization by fine-tuning it on the SAMSum dataset, which is focused on dialogue summarization, and evaluating its performance on both in-domain (SAMSum) and out-of-domain (XSum) datasets. The main objective is to analyze the model's generalization capabilities across different domains and understand its strengths and limitations when applied to diverse textual content.

## Problem Statement

Despite advances in text summarization, models often exhibit reduced performance when applied to domains they were not explicitly trained on. This project aims to quantify and analyze the extent of this performance degradation by fine-tuning a state-of-the-art model on dialogue-based summaries and evaluating its effectiveness on both similar and dissimilar data.

## Objective

The primary objective of this project is to assess the robustness of the DistilBART-cnn-12-6 model in generating summaries across different domains. This will be achieved by:
- Fine-tuning the model on the SAMSum dataset.
- Evaluating its performance on the SAMSum test set (in-domain).
- Evaluating its performance on the XSum test set (out-of-domain).
- Analyzing the results to identify the model’s strengths and weaknesses in cross-domain summarization.

## Datasets

- **SAMSum Dataset:** A dataset comprising human-written summaries of dialogues, used for training and in-domain evaluation. It focuses on summarizing conversational text, which differs from traditional news summarization.
- **XSum Dataset:** A dataset containing news articles paired with one-sentence summaries, used as the out-of-domain evaluation set. It emphasizes extreme abstractive summarization of news content.

## Methodology

### Data Preparation

- The SAMSum dataset was split into training, validation, and test sets.
- The XSum dataset was used solely as a test set for out-of-domain evaluation.

### Model Fine-Tuning

- The DistilBART-cnn-12-6 model was fine-tuned on the SAMSum dataset using the Hugging Face Transformers library. 
- Techniques such as early stopping, learning rate scheduling, and dropout were employed to prevent overfitting.

### Evaluation

- **In-Domain:** Performance was evaluated on the SAMSum test set.
- **Out-of-Domain:** Performance was assessed on the XSum test set.
- Common summarization metrics such as ROUGE were used to quantify the model’s performance across domains.

## Results

### In-Domain (SAMSum Test Set)

- **ROUGE-1:** 0.3989
- **ROUGE-2:** 0.2010
- **ROUGE-L:** 0.3051
- **ROUGE-Lsum:** 0.3053

### Out-of-Domain (XSum Test Subset)

- **ROUGE-1:** 0.2010
- **ROUGE-2:** 0.0382
- **ROUGE-L:** 0.1328
- **ROUGE-Lsum:** 0.1329

## Conclusion

1. **Model's Summarization Tendencies:**
   - The DistilBART-cnn-12-6 model shows a tendency to prioritize initial or topic sentences when summarizing, which is well-suited for news articles where the most important information is often presented upfront. However, this approach can lead to less effective summaries for dialogues, where key information might be spread throughout the conversation rather than concentrated at the beginning.

2. **In-Domain Performance:**
   - The evaluation results demonstrate a significant improvement in the model's performance on dialogue summarization after fine-tuning on the SAMSum dataset. The ROUGE scores indicate that the model's ability to capture key information in dialogue-based content improved notably, underscoring the effectiveness of domain-specific fine-tuning.

3. **Out-of-Domain Performance:**
   - Despite the improvements in dialogue summarization, the model showed only marginal enhancement on the out-of-domain XSum test subset. This suggests that while fine-tuning on SAMSum improved the model's performance in its target domain, the model's ability to generalize to more extreme extractive summarization tasks, like those in XSum, remains limited.

## Future Work

To further improve the model's robustness across different domains, future work could explore multi-domain training, domain adaptation techniques, or fine-tuning on a combination of datasets. Additionally, developing more advanced metrics to better capture the quality of summaries across diverse contexts would be beneficial.

## Citation

If you use this code or find this work helpful in your research, please consider citing the following sources:

**SAMSum Dataset:**
```
@inproceedings{gliwa-etal-2019-samsum,
    title = "{SAMS}um Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization",
    author = "Gliwa, Bogdan  and
      Mochol, Iwona  and
      Biesek, Maciej  and
      Wawer, Aleksander",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5409",
    doi = "10.18653/v1/D19-5409",
    pages = "70--79"
}
```

**XSum Dataset:**
```
@article{Narayan2018DontGM,
  title={Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization},
  author={Shashi Narayan and Shay B. Cohen and Mirella Lapata},
  journal={ArXiv},
  year={2018},
  volume={abs/1808.08745}
}
```

**DistilBART-cnn-12-6 Model:**
Please refer to the [Hugging Face Model Hub](https://huggingface.co/sshleifer/distilbart-cnn-12-6) for citation details.

## Licensing Information

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.

## Contributions

Thanks to @cccntu, @thomwolf, @lewtun, @mariamabarham, @jbragg, @lhoestq, and @patrickvonplaten for their contributions to the datasets and model.

## Acknowledgements

We would like to acknowledge the Hugging Face team for providing the tools and infrastructure that made this project possible. Special thanks to the contributors of the SAMSum and XSum datasets, and to the developers of the DistilBART model. 
