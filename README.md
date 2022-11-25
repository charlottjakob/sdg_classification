# SDG Classification

This repository includes the code for classifying SDG in companies' sustainability reports as explained in a master thesis.
The following steps need to be done in oder to recreate the results.

Using the training, test and validation sets in folder /data it is possible to start at Step 9: Train and test SVM


## Steps

1. Scraping: Run all spiders
    * scripts: /scraping/scraping/spiders
    * output: 
        * un_data, 
        * scholar_data, 
        * wikipedia_data, 
        * un_global_compact_data_1_500, 
        * un_global_compact_data_500_1500


2. Manuall Labeling of 250 sustainability reports
    * -> manually open sustainability reports and collect document, page and sdgs in google sheet
    * output: 
        * image_labels_manual

3. Cut reports into pages with and pages without SDGs
    * script: data_preperation/images/cut_report_pages.py
    * input:
        * image_labels_manual.csv
        * sustainability_reports_1_500/
    * output:
        * report_pages_nosdg/file_names.csv
        * report_pages_nosdg/
        * image_test_labels.csv
        * image_test/

4. Create selfmaid training data  (Random Watermarks on top of Pages withoug SDGs)
    * script: data_preperation/images/creat_image_train_set.py
    * input:
        * report_pages_nosdg/
        * report_pages_nosdg/file_names.csv
    * output:
        * image_train/
        * image_train_labels.csv

5. Train Image Classifier (selfmad data + 300 pages with SDGs = Training, Rest of the pages with SDGs = Testing Data)
    * script: training/image/image_main.py
    * input: 
        * image_train/
        * image_train_labels.csv
        * image_test/ 
        * image_test_labels.csv
    * output:
        * image.model
        * image_history.txt      


6. Approve 2000 sustainability reports (extractable & english language)
    * script: data_preperation/text/approve_sr_before_labeling.py
    * input:
        * sustainability_reports_500_1500/
        * un_global_compact_data_500_1500.csv
    * output:
        * reports_approval.csv

7. Predict approved sustainability reports
    * script: training/image/image_prediction.py
    * input:
        * sustainability_reports_500_1500/
        * reports_approval.csv
    * output:
        * reports_predictions.csv

8. Extract Text of pages with positve Outputs
    * script: data_preperation/text/extract_predicted_pages.py
    * input:
        * sustainability_reports_500_1500/
        * reports_predictions.csv 
    * output:
        * sr_data.csv


9. Clean, Balance Trainingsets and Clean Testingset for Text Classification
    * script: script: /main
    * input:
        * un_data, 
        * scholar_data, 
        * wikipedia_data
        * sr_data.csv
    * output:
        * text_train_1.csv
        * text_train_2.csv
        * text_test_1.csv
        * text_test_2.csv

10. Create data from Further Pre-training
    * script: data_preperation/text/create_pretraining_data.py
    * input:
        * sustainability_reports_500_1500/ 
    * output:
        * data/text_domain.csv

11. Create Further-Pretrained model for each transformer
    * script: training/text/transformers/pretraining.py
    * input:
        * data/text_domain.csv
    * output:
        * training/text/transformers/models/pretrained_*

12. Train and test SVM
    * script: script: training/text/SVM.py
    * input:
        * text_train_1.csv
        * text_train_2.csv
    * output:
        * data/text_test_results/word2vec_svm.csv
        * data/text_test_results/glove_svm.csv

13. Train all Transformers
    * script: script: training/text/transformers/training
    * input:
        * text_train_1.csv
        * text_train_2.csv
    * output:
        * data/histories/*
        * training/text/transformers/models/*

14. Validate Transformers
    * script: training/text/transformers/testing
    * input:
        * text_test.csv
        * training/text/transformers/models/ 
    * output:
        * data/text_test_results/

15. Plot Results with 
    * script: plotting.ipynb
    * input:
        * data/text_test_results/
        * data/histories/
    * output: 
        * data/plot_images

