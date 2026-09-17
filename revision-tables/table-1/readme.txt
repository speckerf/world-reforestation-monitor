CHANGELOG September 2026: 
Table 1 only shows R2 and RMSE for S2BIOPHYS and SL2P. The full table is now shown in Table S4, which also shows GROUNDED-EO GPR model performance for FAPAR.  


--------------

Please see the following function for how to obtain the metrics:


file: train_pipeline/predict_insitu_comparison.py
function: create_revisions_table1()

Which creates:
- revision_table1_model_comparison.csv
- revision_table1_model_comparison.tex

Then, the tex file is input for manual editing of the table (e.g. highlighting the best score)
