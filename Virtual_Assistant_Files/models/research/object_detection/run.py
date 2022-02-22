python model_main_tf2.py --pipeline_config_path=training_detector\pipeline.config --model_dir=training_detector --alsologtostderr
python exporter_main_v2.py --trained_checkpoint_dir=training_detector --pipeline_config_path=training_detector\pipeline.config --output_directory inference_graph 
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record