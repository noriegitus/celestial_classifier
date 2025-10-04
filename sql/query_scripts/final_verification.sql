-- ===== CHECK THE HEALTH OF YOUR DATABASE AND THAT ALL DATA IS CORRECTLY IMPORTED =====

SELECT
    m.architecture || ' ' || m.version AS model_name,
    i.source,
    p.is_correct,
    COUNT(*) AS total
FROM
    Prediction p
JOIN
    Model m ON p.model_id = m.model_id
JOIN
    Image i ON p.image_id = i.image_id
GROUP BY
    model_name, i.source, p.is_correct
ORDER BY
    model_name, i.source, p.is_correct;