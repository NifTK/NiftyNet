# Configuration file
*[This folder](../config) presents a few examples of configuration files for NiftyNet
[applications](#../niftynet/application/).*

Please check out the configuration specifications at

[http://niftynet.readthedocs.io/en/dev/config_spec.html](http://niftynet.readthedocs.io/en/dev/config_spec.html)


#Evaluation
For evaluation of the output of an application against some available ground 
truth, an EVALUATION section must be present. An example of an evaluation 
config file is available for different application with the suffix _eval

The evaluation command is run as:
net_run.py evaluation -a APPLICATIONPATH -c CONFIGFILEPATH 
Example net_run.py evaluation -a niftynet.applications
.segmentation_application.SegmentationApplication -c 
config/default_multimodal_segmentation_eval.ini
In order to run the evaluation, the input sources section must contain the 
details on
* The ground truth against which to compare (label in case of the segmentation)
* The corresponding files to evaluate (inferred)

The final evaluation file is saved in the folder indicated as input of the 
field model_dir in the section [SYSTEM] under the form of a csv file with 
indication of subject id (label if relevant) and the calculated metrics as 
columns.

The evaluation section ([EVALUATION]) must contain
* save_csv_dir = Path where to save the csv output
* evaluations = List of metrics of evaluation to be calculated presented as a
 string separated by commas (e.g dice,jaccard,n_pos_ref,n_pos_seg). Lists of 
 possible evaluations metrics per application are available in
 http://niftynet.readthedocs.io/en/dev/niftynet.evaluation.regression_evaluations.html
 http://niftynet.readthedocs.io/en/dev/niftynet.evaluation
 .segmentation_evaluations.html
 http://niftynet.readthedocs.io/en/dev/niftynet.evaluation
 .classification_evaluations.html
 * evaluation_units = foreground, label or cc. Describe how the evaluation 
 should be performed in the case of segmentation mostly (foreground = only one 
 label, label (metrics per label), cc (metrics per connected component) ) 
 More on this topic can be found at http://niftynet.readthedocs.io/en/dev/niftynet.evaluation.segmentation_evaluations.html
 
 

[EVALUATION]

save_csv_dir = ./NiftyNetTestEval
evaluations = dice,jaccard,false_positive_rate,positive_predictive_values,n_pos_ref,n_pos_seg
evaluation_units = foreground

[APPLICATION SPECIFIC]
The fields defined as input sources must be specified to be used in the 
evaluation and notably for segmentation application (label inferred) and 
(output inferred) for the regression