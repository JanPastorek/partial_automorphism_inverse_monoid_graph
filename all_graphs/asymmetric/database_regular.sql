-- record asymmetric graphs
SELECT number_of_vertices, regular, max(asymmetric_depth) + 1 as "max(asymmetric_depth)"  FROM graphs
WHERE regular IS NOT NULL
GROUP BY number_of_vertices, regular;