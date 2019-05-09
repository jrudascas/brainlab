from scipy.stats import mannwhitneyu

FLID = [
520,
512,
472,
580,
543,
412,
580,
474,
592,
579]
FLII = [
617,
610,
574,
623,
577,
464,
645,
511,
571,
564
]


print(mannwhitneyu(FLID, FLII))