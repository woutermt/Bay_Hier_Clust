
f = "Raw_data_files/rosalind_dna.txt"

with open(f, 'r') as file:
    dna = file.read()

dna_dict = {}
for i in range(len(dna)):
    if dna[i] not in dna_dict:
        dna_dict[dna[i]] = 1
    else:
        dna_dict[dna[i]] += 1


for key, value in dna_dict.items():
    print(key, value)


