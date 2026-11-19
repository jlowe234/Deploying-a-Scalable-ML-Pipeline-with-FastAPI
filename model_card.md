# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Developed by Jonathan Lowe for WGU 501: Machine Learning DevOps.
- The model is a Logistic Regression model trained to classify people into high (>=50K) and low (<50K) earner groups.

## Intended Use

- Intended to be used for educational purposes, specifically around training and consuming ML models from an API.
- This is not suited to support demographic inference, population-level analysis, social science research, or policy development.

## Training Data

- The model was trained using UC Irvine's [Census Income 1996](https://archive.ics.uci.edu/dataset/20/census+income) dataset, training split.

## Evaluation Data

- Evaluation was performed using the UC Irvine's [Census Income 1996](https://archive.ics.uci.edu/dataset/20/census+income) dataset, test split.

## Metrics

- Precision or how often the model's positive prediction is correct, for this model was 0.7452
- Recall or the number of positive cases the model successfully identified, for this model was 0.6056
- F1 Score, which is the balance between precision and recall, with this model was 0.6682

## Ethical Considerations

- Evaluation of this model across demographic groups identified significant differences in precision and recall between racial groups.
- This model also has disparities in recall for female populations.
- Both of these findings show that the fairness of this model may not be very high.

## Caveats and Recommendations

- The training dataset had underrepresentation across some demographic groups, like Armed Forces workers, American Indian, and Asian/Pacific Islanders. These groups may result in less accurate performance.
- It would be recommended to collect more data among the underrepresented groups and refine the model further.