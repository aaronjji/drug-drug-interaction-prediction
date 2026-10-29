# Text-leakage screen (camera-ready), filter level: generic

- Drugs with a PubChem description: 1501/1706; 4607 sentences.
- Descriptions with generic co-administration / interaction-mechanism wording (combination use, CYP450, P-glycoprotein, enzyme induction, contraindication, ...): 71 (4.7%).
- Descriptions with >=1 interaction-type term (template effect terms + class synonyms): 778 (51.8%).
- Filter removed 78 sentences (1.7%) and 2.7% of words; 71 descriptions changed, 0 emptied (-> missing-text token); 1501 drugs keep some text.

## Pairs whose own interaction type's term appears in a description

| split | n pairs | types with a term | Drug2 desc. matches (full) | either (full) | either (filtered) |
|---|---|---|---|---|---|
| transductive_train | 134264 | 100.0% | 6.6% | 12.9% | 12.6% |
| transductive_test | 38366 | 100.0% | 6.6% | 13.0% | 12.7% |
| inductive_train | 99372 | 100.0% | 6.9% | 12.9% | 12.7% |
| s1 | 57792 | 100.0% | 6.1% | 12.6% | 12.4% |
| s2 | 6617 | 100.0% | 5.7% | 13.7% | 13.6% |

## Text coverage by inductive drug group (full -> filtered)

- train: 88.0% -> 88.0%
- val: 87.6% -> 87.6%
- test: 88.0% -> 88.0%

## Examples (before -> after)

**DB04571**

- before: Trioxsalen is 7H-Furo[3,2-g]chromen-7-one in which positions 2, 5, and 9 are substituted by methyl groups. Like other psoralens, trioxsalen causes photosensitization of the skin. It is administered orally in conjunction with UV-A for phototherapy treatment of vitiligo. After photoactivation it creates interstrand cross-links in DNA, inhibiting DNA synthesis and cell division, and can lead to cell injury; recovery from the cell injury may be followed by increased melanisation of the epidermis. It has a role as a photosensitizing agent and a dermatologic drug.
- after: Trioxsalen is 7H-Furo[3,2-g]chromen-7-one in which positions 2, 5, and 9 are substituted by methyl groups. Like other psoralens, trioxsalen causes photosensitization of the skin. After photoactivation it creates interstrand cross-links in DNA, inhibiting DNA synthesis and cell division, and can lead to cell injury; recovery from the cell injury may be followed by increased melanisation of the epidermis. It has a role as a photosensitizing agent and a dermatologic drug.

**DB00855**

- before: 5-aminolevulinic acid is the simplest delta-amino acid in which the hydrogens at the gamma position are replaced by an oxo group. It is metabolised to protoporphyrin IX, a photoactive compound which accumulates in the skin. Used (in the form of the hydrochloride salt)in combination with blue light illumination for the treatment of minimally to moderately thick actinic keratosis of the face or scalp. It has a role as an antineoplastic agent, a photosensitizing agent, a dermatologic drug, a prodrug, a mouse metabolite, a Saccharomyces cerevisiae metabolite, a plant metabolite, an Escherichia coli metabolite and a human metabolite. It is a delta-amino acid and a 4-oxo monocarboxylic acid. It is functionally related to a 4-oxopentanoic acid. It is a conjugate base of a 5-ammoniolevulinic acid. It is a conjugate acid of a 5-aminolevulinate. It is a tautomer of a 5-ammoniolevulinate.
- after: 5-aminolevulinic acid is the simplest delta-amino acid in which the hydrogens at the gamma position are replaced by an oxo group. It is metabolised to protoporphyrin IX, a photoactive compound which accumulates in the skin. It has a role as an antineoplastic agent, a photosensitizing agent, a dermatologic drug, a prodrug, a mouse metabolite, a Saccharomyces cerevisiae metabolite, a plant metabolite, an Escherichia coli metabolite and a human metabolite. It is a delta-amino acid and a 4-oxo monocarboxylic acid. It is functionally related to a 4-oxopentanoic acid. It is a conjugate base of a 5-ammoniolevulinic acid. It is a conjugate acid of a 5-aminolevulinate. It is a tautomer of a 5-ammoniolevulinate.

**DB00908**

- before: Quinidine is a cinchona alkaloid consisting of cinchonine with the hydrogen at the 6-position of the quinoline ring substituted by methoxy. It has a role as an alpha-adrenergic antagonist, an antimalarial, an anti-arrhythmia drug, a sodium channel blocker, a muscarinic antagonist, a P450 inhibitor, a potassium channel blocker, an EC 3.6.3.44 (xenobiotic-transporting ATPase) inhibitor, an EC 1.14.13.181 (13-deoxydaunorubicin hydroxylase) inhibitor and a drug allergen. It derives from a hydride of a cinchonan.
- after: Quinidine is a cinchona alkaloid consisting of cinchonine with the hydrogen at the 6-position of the quinoline ring substituted by methoxy. It derives from a hydride of a cinchonan.

**DB00238**

- before: Nevirapine is a dipyridodiazepine that is 5,11-dihydro-6H-dipyrido[3,2-b:2',3'-e][1,4]diazepine which is substituted by methyl, oxo, and cyclopropyl groups at positions 4, 6, and 11, respectively. A non-nucleoside reverse transcriptase inhibitor with activity against HIV-1, it is used in combination with other antiretrovirals for the treatment of HIV infection. It has a role as an antiviral drug and a HIV-1 reverse transcriptase inhibitor. It is a member of cyclopropanes and a dipyridodiazepine.
- after: Nevirapine is a dipyridodiazepine that is 5,11-dihydro-6H-dipyrido[3,2-b:2',3'-e][1,4]diazepine which is substituted by methyl, oxo, and cyclopropyl groups at positions 4, 6, and 11, respectively. It has a role as an antiviral drug and a HIV-1 reverse transcriptase inhibitor. It is a member of cyclopropanes and a dipyridodiazepine.

