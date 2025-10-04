-- ======== SCRIPT TO PASTE IN psql COMMAND LINE. THIS WONT WORK ON pgAdmin ==========

-- Creates a temporary table to hold the data from each CSV file before inserting it.
CREATE TEMP TABLE temp_predictions (
    filename VARCHAR(255),
    true_label VARCHAR(50),
    predicted_label VARCHAR(50),
    confidence NUMERIC(10,9),
    is_correct BOOLEAN
);

-- Block 1: CNN v1
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/cnn_v1_predictions_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 1, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 2: CNN v2
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/cnn_v2_predictions_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 2, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 3: ResNet18 v1
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/resnet18_v1_predictions_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 3, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 4: ResNet18 v2
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/resnet18_v2_predictions_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 4, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 5: ResNet18 v3
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/resnet18_v3_predictions_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 5, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 6: External CNN v1
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/cnn_v1_predictions_outside_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 1, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 7: External CNN v2
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/cnn_v2_predictions_outside_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 2, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 8: External ResNet18 v1
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/resnet18_v1_predictions_outside_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 3, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 9: External ResNet18 v2
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/resnet18_v2_predictions_outside_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 4, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Block 10: External ResNet18 v3
\copy temp_predictions FROM 'C:/Users/nori/ProgrammingProjects/celestial_classifier/outputs/processed_data/resnet18_v3_predictions_outside_ENRICHED.csv' DELIMITER ',' CSV HEADER;
INSERT INTO Prediction (image_id, model_id, predicted_label, is_correct, confidence) SELECT i.image_id, 5, t.predicted_label::galaxy_class, t.is_correct, t.confidence FROM temp_predictions t JOIN Image i ON t.filename = i.filename;
TRUNCATE temp_predictions;

-- Final Verification Query
SELECT
    m.architecture,
    m.version,
    i.source,
    COUNT(p.prediction_id) as total_predictions
FROM
    Prediction p
JOIN
    Model m ON p.model_id = m.model_id
JOIN
    Image i ON p.image_id = i.image_id
GROUP BY
    m.model_id, m.architecture, m.version, i.source
ORDER BY
    m.model_id, i.source;