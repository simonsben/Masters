# Sarcasm

The identification of sarcasm would be very beneficial since its use creates a reversal in the sentiment expressed.
Therefore, without identifying it it could cause a sarcastic statement to become either a false positive or a false negative (depending on the original polarity of the statement).

## Findings in literature

Not surprisingly, this is a topic that has been considered by many NLP/linguistics researchers.
Generally, it has been seen that automatic identification is **very** hard.
This is because much of sarcasm can be seen to context and user specific [[Paper](https://arxiv.org/pdf/1610.08815.pdf), [Summary](https://medium.com/dair-ai/detecting-sarcasm-with-deep-convolutional-neural-networks-4a0657f79e80)].
The paper showed that if these aspects were considered an accuracy of ~95% is achievable.
However, if the test data is different (i.e. different sources) from the training set this accuracy was reduced to ~75% (or worse).
