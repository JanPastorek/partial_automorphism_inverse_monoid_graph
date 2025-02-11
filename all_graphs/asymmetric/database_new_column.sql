-- record asymmetric graphs
ALTER TABLE graphs
  RENAME COLUMN graph6 TO canonical_label;
