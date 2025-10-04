
-- Inserting both models in table 'Model'
-- Assuming model_id=1 for'CNN' and model_id=2 for 'ResNet18'
INSERT INTO Model (architecture, version, description) VALUES
('CNN', 'v1', 'Modelo CNN desde cero primera version'),
('CNN', 'v2', 'Modelo CNN desde cero con 3 capas convolucionales.'),
('ResNet18', 'v1', 'Primera versión de ResNet18 sin aumento de datos'),
('ResNet18', 'v2', 'Segunda versión de ResNet18 con ajustes de hiperparámetros'),
('ResNet18', 'v3', 'Modelo ResNet18 pre-entrenado con transfer learning.');

-- Inserting the metrics of each model on table 'Evaluation'
INSERT INTO Evaluation (model_id, accuracy, precision, recall, f1_score) VALUES
(1, 0.8060, 0.8819, 0.8165, 0.8480), -- Métricas para CNN v1 (ID=1)
(2, 0.8116, 0.8771, 0.8323, 0.8541), -- Métricas para CNN v2 (ID=2)
(3, 0.7941, 0.8445, 0.8449, 0.8447), -- Métricas para ResNet18 v1 (ID=3)
(4, 0.7526, 0.7841, 0.8648, 0.8225), -- Métricas para ResNet18 v2 (ID=4)
(5, 0.7540, 0.8493, 0.7644, 0.8046); -- Métricas para ResNet18 v3 (ID=5)

SELECT * FROM Model;
SELECT * FROM Evaluation;