### limitations_and_non_usage.md

### recognizing limits as an integral part of the system

Any decision support system based on statistical models relies on a set of assumptions, often implicit. In the case of churn, these assumptions concern both the stability of customer behavior, the representativeness of the available data, and the model’s ability to capture signals that are genuinely relevant for action. It would be misleading to assume that these assumptions are always satisfied, or that they can be ignored without consequences.

In this project, the system’s limitations are not treated as flaws to be concealed, but as constitutive elements of its responsible use. They condition not only the level of trust that can be placed in the results produced, but also the situations in which the system should simply not be used.

### limits related to available data

The dataset used presents several structural limitations. It is a static dataset, which does not allow tracking customer evolution over time nor observing the impact of any retention actions that may have been implemented. This lack of a temporal dimension restricts the scope of the predictions, particularly when it comes to anticipating complex or progressive behaviors.

Moreover, certain pieces of information that could be decisive for decision-making are not present in the data. Qualitative elements, such as recent interactions with customer service, exceptional events, or personal changes, are by nature absent from this type of dataset. The model can therefore reason only on the basis of a partial view of the customer’s situation.

It should also be noted that the data reflect a specific context, both commercially and geographically. Any direct transposition of the system to another environment, without prior adaptation, would introduce a significant risk of misinterpretation.

### limits related to the model and learning process

The classification model used in this project is deliberately simple and interpretable. This choice facilitates system analysis and governance, but it also implies certain limitations in terms of representational capacity. Complex non-linear relationships or subtle contextual effects may not be captured satisfactorily.

Even when acceptable metrics are observed, it must be recalled that performance measured on a test set does not guarantee similar behavior in real-world conditions. Data distributions may evolve, and learned patterns may lose their relevance. In this context, excessive confidence in the scores produced constitutes a risk in itself.

It might be tempting to respond to these limitations by increasing model complexity. However, this approach raises other issues, particularly with respect to interpretability and control. The project deliberately chooses not to pursue this path, accepting the trade-offs associated with simpler models.

### situations of explicit non-use of the system

Certain situations explicitly justify non-use of the system. This is notably the case when input data are clearly incomplete, outdated, or inconsistent. In such a context, producing a risk estimate would create an illusion of precision with no real foundation.

Similarly, when major changes occur in the commercial offering, pricing policy, or contract structure, the assumptions on which the model is based may become invalid. Until retraining and re-evaluation have been carried out, use of the system should be suspended.

Individual atypical cases must also be considered, where observed profiles deviate strongly from those present in the training data. In these situations, the model is likely to produce unreliable scores. The logic adopted here is therefore to favor human analysis, without relying on the predictive signal.

### risks of abusive interpretation of scores

A frequently encountered limitation in predictive systems concerns the interpretation of scores as objective probabilities. A high score may be perceived as a certainty of churn, whereas it represents only a conditional estimate, dependent on the model and the data used.

This confusion can lead to excessively confident decisions or to an abusive standardization of actions. The project therefore emphasizes the need to contextualize scores and to consider them as relative indicators rather than as verdicts.

From an operational perspective, this implies that system users must be trained in reading and interpreting the results, and encouraged to question the recommendations produced rather than applying them mechanically.

### accepting uncertainty as a condition for responsible use

One of the implicit conclusions of this work is that uncertainty cannot be eliminated from a problem such as churn. Attempting to reduce it at all costs, through technical artifices or reassuring metrics, risks instead displacing it and making it less visible.

Recognizing the system’s limitations, defining rules for non-use, and maintaining a central role for human intervention are ways of managing this uncertainty rather than denying it. This stance may appear cautious, even conservative, but it is more consistent with the requirements of responsible AI use in real-world contexts.

In this sense, the system presented here is not designed as a definitive solution, but as an improvable tool whose value depends as much on its acknowledged limits as on its predictive capabilities. It is within this somestimes fragile balance that its relevance resides.
