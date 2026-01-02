## Challenges & Solutions

### What was the hardest part of this assignment?

The hardest part of this assignment was not implementing the model, but correctly understanding and handling the evaluation process. Initially, it seemed reasonable to compute quantitative metrics such as MSE directly on the test set because test annotations were provided. However, after closely inspecting the annotation files, it became clear that the test set did not contain ground-truth trajectories. This required a shift in thinking—from treating the test set as a quantitative benchmark to using it only for qualitative evaluation and relying on a validation split from the training data for numerical metrics. Recognizing and correcting this misunderstanding was the most challenging and educational part of the assignment.

---

### Describe a specific bug you encountered and how you debugged it

A significant bug encountered during evaluation was a `KeyError: 'path'` that occurred when iterating over the test DataLoader. The error was triggered when the code attempted to access the `"path"` field in test annotations using the same dataset loader designed for training data.

To debug this issue, the first step was to carefully read the error message and identify where the failure occurred. I then manually opened several test annotation JSON files and compared them with training annotations. This comparison revealed that, unlike the training data, test annotations did not include the `"path"` field.

The root cause was an incorrect assumption that both datasets shared the same schema. The fix involved separating the evaluation logic: quantitative evaluation (MSE) was performed only on a validation split of the training data, while the test set was used strictly for inference and qualitative visualization. This debugging process reinforced the importance of validating dataset structure before designing evaluation pipelines.

## Experiment Notebook
I have also submitted the experiment notebook for the reference.

## Test Images
I have also add the images of testing the on test images for predicting the path with it's visualizations.
