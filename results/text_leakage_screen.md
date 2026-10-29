# Text-leakage screen (camera-ready), filter level: all

- Drugs with a PubChem description: 1501/1706; 4607 sentences.
- Descriptions with generic co-administration / interaction-mechanism wording (combination use, CYP450, P-glycoprotein, enzyme induction, contraindication, ...): 71 (4.7%).
- Descriptions with >=1 interaction-type term (template effect terms + class synonyms): 778 (51.8%).
- Filter removed 1167 sentences (25.3%) and 30.6% of words; 829 descriptions changed, 119 emptied (-> missing-text token); 1382 drugs keep some text.

## Pairs whose own interaction type's term appears in a description

| split | n pairs | types with a term | Drug2 desc. matches (full) | either (full) | either (filtered) |
|---|---|---|---|---|---|
| transductive_train | 134264 | 100.0% | 6.6% | 12.9% | 0.0% |
| transductive_test | 38366 | 100.0% | 6.6% | 13.0% | 0.0% |
| inductive_train | 99372 | 100.0% | 6.9% | 12.9% | 0.0% |
| s1 | 57792 | 100.0% | 6.1% | 12.6% | 0.0% |
| s2 | 6617 | 100.0% | 5.7% | 13.7% | 0.0% |

## Text coverage by inductive drug group (full -> filtered)

- train: 88.0% -> 80.7%
- val: 87.6% -> 82.9%
- test: 88.0% -> 81.3%

## Examples (before -> after)

**DB04571**

- before: Trioxsalen is 7H-Furo[3,2-g]chromen-7-one in which positions 2, 5, and 9 are substituted by methyl groups. Like other psoralens, trioxsalen causes photosensitization of the skin. It is administered orally in conjunction with UV-A for phototherapy treatment of vitiligo. After photoactivation it creates interstrand cross-links in DNA, inhibiting DNA synthesis and cell division, and can lead to cell injury; recovery from the cell injury may be followed by increased melanisation of the epidermis. It has a role as a photosensitizing agent and a dermatologic drug.
- after: Trioxsalen is 7H-Furo[3,2-g]chromen-7-one in which positions 2, 5, and 9 are substituted by methyl groups. After photoactivation it creates interstrand cross-links in DNA, inhibiting DNA synthesis and cell division, and can lead to cell injury; recovery from the cell injury may be followed by increased melanisation of the epidermis.

**DB00855**

- before: 5-aminolevulinic acid is the simplest delta-amino acid in which the hydrogens at the gamma position are replaced by an oxo group. It is metabolised to protoporphyrin IX, a photoactive compound which accumulates in the skin. Used (in the form of the hydrochloride salt)in combination with blue light illumination for the treatment of minimally to moderately thick actinic keratosis of the face or scalp. It has a role as an antineoplastic agent, a photosensitizing agent, a dermatologic drug, a prodrug, a mouse metabolite, a Saccharomyces cerevisiae metabolite, a plant metabolite, an Escherichia coli metabolite and a human metabolite. It is a delta-amino acid and a 4-oxo monocarboxylic acid. It is functionally related to a 4-oxopentanoic acid. It is a conjugate base of a 5-ammoniolevulinic acid. It is a conjugate acid of a 5-aminolevulinate. It is a tautomer of a 5-ammoniolevulinate.
- after: 5-aminolevulinic acid is the simplest delta-amino acid in which the hydrogens at the gamma position are replaced by an oxo group. It is a delta-amino acid and a 4-oxo monocarboxylic acid. It is functionally related to a 4-oxopentanoic acid. It is a conjugate base of a 5-ammoniolevulinic acid. It is a conjugate acid of a 5-aminolevulinate. It is a tautomer of a 5-ammoniolevulinate.

**DB01600**

- before: Tiaprofenic acid is an aromatic ketone that is thiophene substituted at C-2 by benzoyl and at C-4 by a 1-carboxyethyl group. It has a role as a non-steroidal anti-inflammatory drug and a drug allergen. It is a monocarboxylic acid, a member of thiophenes and an aromatic ketone.
- after: Tiaprofenic acid is an aromatic ketone that is thiophene substituted at C-2 by benzoyl and at C-4 by a 1-carboxyethyl group. It is a monocarboxylic acid, a member of thiophenes and an aromatic ketone.

**DB11630**

- before: Temoporfin is a member of chlorins. It has a role as a photosensitizing agent.
- after: Temoporfin is a member of chlorins.

**DB00670**

- before: Pirenzepine is a pyridobenzodiazepine. It has a role as a muscarinic antagonist, an anti-ulcer drug and an antispasmodic drug.
- after: (empty -> missing-text token)

**DB01116**

- before: Trimethaphan is a complex heterocyclic sulfonium compound with an imidazolium core, used to treat hypertension. It has a role as a vasodilator agent, an antihypertensive agent, a nicotinic antagonist and an anaesthesia adjuvant.
- after: (empty -> missing-text token)

