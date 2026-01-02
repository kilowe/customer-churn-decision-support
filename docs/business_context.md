### business_context.md

understanding churn beyond prediction

Customer churn is often presented as a relatively straightforward classification problem: a customer leaves or stays. This view, while technically convenient, tends to obscure the diversity of real-world situations that underlie a cancellation. In an operational context, not all churn events are equivalent, and above all, not all are equally predictable or actionable.

From a business perspective, it is useful to distinguish between several forms of churn. Some departures stem from factors largely exogenous to the service itself: relocations, sudden financial constraints, or decisions imposed by a third party. Others, by contrast, appear more closely linked to the customer experience, the perceived value of the service, or frictions accumulated over time. This distinction, while simplificatory, has important consequences for how a predictive system can be used.

In the context of a telecom operator, churn represents not only the loss of a customer, but also the loss of future revenue, sometimes combined with indirect costs related to acquiring new customers. In this setting, it is tempting to try to identify as early as possible the customers who are likely to leave. However, overly aggressive anticipation can lead to ineffective or even counterproductive actions if it is based on weak or misinterpreted signals.

The Telco Customer Churn dataset used in this project partially reflects this complexity. It aggregates heterogeneous information: contractual data, subscribed services, billing elements, as well as more general variables related to tenure or payment method. Taken in isolation, none of these attributes is sufficient to explain a cancellation. Their value lies instead in the interactions between them and in the patterns they may suggest when analyzed jointly.

However, the scope of these data must be qualified. The dataset is static and does not capture the temporal dimension of churn, nor the actions actually taken by the company prior to a cancellation. As such, it does not allow direct observation of the effects of a retention policy. The model learned from these data can therefore only be understood as an approximation, useful for exploring certain hypotheses, but insufficient to dictate decisions without caution.

This is precisely why this project does not consider prediction as an end in itself. The score produced by a churn model is only one signal among others, whose interpretation depends heavily on the business context in which it is used. The same level of risk will not have the same implications depending on the type of customer, the nature of their contract, or the margin associated with their profile.

One might be tempted to treat churn as a purely technical problem, seeking to continuously improve the performance of a classifier. From an empirical standpoint, this approach quickly reveals its limitations. It tends to ignore organizational constraints, actual operational capacities, and human responsibility in decision-making. By contrast, viewing churn as a decision problem makes it possible to reposition the model within a broader system, where it serves to inform a possible action without ever replacing it.

From this perspective, the business context is not merely a preamble to the technical work. It guides the choices made, sets boundaries, and conditions the use of the results produced. It is from this reading that the system presented in this project was designed—not as an autonomous tool, but as a decision support component, integrated into an existing process and subject to explicit rules of use.
