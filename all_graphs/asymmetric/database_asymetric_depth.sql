-- record asymmetric graphs
SELECT number_of_vertices, max(asymmetric_depth) + 1 FROM graphs
GROUP BY number_of_vertices;