clean:
	rm -rf section_4/results section_5/processed_data

part_a:
	mkdir section_4/results
	python3.8 section_4/kinbank_classify_sib_uncle_paper_03.py
	
part_b:
	mkdir section_5/processed_data
	RScript section_5/make_distancematrix.R
	mkdir section_5/results
	
test:
	mkdir section_5/results
	python3.8 section_5/hdbscan_umap2.py
	#RScript section_5/make-tsne-graph.R
	
test_clean:
	rm -rf section_5/results