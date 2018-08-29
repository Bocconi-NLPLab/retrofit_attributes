# Increasing In-Class Similarity by Retrofitting Embeddings with Demographic Information

## Abstract
Most text-classification approaches represent the input based on textual features, either feature-based or continuous.
However, this ignores strong non-linguistic similarities like homophily: people within a demographic group use language more similar to each other than to non-group members.
We use homophily cues to retrofit text-based author representations with non-linguistic information, and introduce a trade-off parameter.
This approach increases in-class similarity between authors, and improves classification performance by making classes more linearly separable. We evaluate the effect of our method on two author-attribute prediction tasks with various training-set sizes and parameter settings.
We find that our method can significantly improve classification performance, especially when the number of labels is large and limited labeled data is available. It is potentially applicable as preprocessing step to any text-classification task.

## References

The paper appeared at EMNLP 2018:
* **Dirk Hovy and Tommaso Fornaciari**. 2018. *Improving Author Attribute Prediction by Retrofitting Linguistic Representations with Homophily*. In Proceedings of EMNLP.

```bib
@inproceedings{HovyFornaciari2018increasing,
  title={{Increasing In-Class Similarity by Retrofitting Embeddings with Demographic Information}},
  author={Hovy, Dirk and Fornaciari, Tommaso},
  booktitle={Proceedings of the 2018 conference on Empirical Methods in Natural Language Processing},
  year={2018}
}
```
