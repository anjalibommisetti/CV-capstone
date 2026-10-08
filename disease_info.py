"""
disease_info.py
Knowledge Base for PlantVillage Dataset (38 Classes across 14 Crops).

Provides structured crop names, disease names, health status,
recommended treatments, and prevention tips for agricultural decision support.
"""

PLANT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

DISEASE_DETAILS = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "is_healthy": False,
        "description": "Fungal infection (Venturia inaequalis) causing olive-green to black velvety spots on leaves and fruit.",
        "treatment": "Apply sulfur or copper-based fungicides during bud break. For severe cases, use captan or myclobutanil fungicides.",
        "prevention": "Rake and destroy fallen leaves in autumn, prune tree canopies to improve air circulation, and choose scab-resistant apple varieties."
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot",
        "is_healthy": False,
        "description": "Fungal disease (Botryosphaeria obtusa) causing 'frog-eye' leaf spots, fruit decay, and bark cankers.",
        "treatment": "Prune out dead wood, mummified fruits, and infected twigs. Apply broad-spectrum copper fungicides or captan during bloom.",
        "prevention": "Maintain tree vigor with balanced fertilizer, disinfect pruning tools with 70% alcohol, and promptly remove damaged fruit."
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "is_healthy": False,
        "description": "Gymnosporangium fungal disease causing bright orange-yellow spots on upper leaf surfaces.",
        "treatment": "Apply fungicides containing myclobutanil or mancozeb when cedar galls swell in early spring.",
        "prevention": "Remove nearby eastern red cedar or juniper trees within 500 meters if feasible; plant rust-resistant cultivars."
    },
    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "The leaf shows vigorous green coloration with no visible signs of pathogenic infection.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Continue regular balanced fertilization, proper drip irrigation, and seasonal pruning."
    },
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Healthy blueberry foliage with uniform color and normal vascular structure.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Maintain soil pH between 4.5 and 5.2, apply organic pine mulch, and avoid overwatering."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "description": "Fungal disease (Podosphaera clandestina) resulting in powdery white circular fungal patches on leaves.",
        "treatment": "Spray wettable sulfur, potassium bicarbonate, or neem oil at first sign of white powder.",
        "prevention": "Prune inner branches to maximize sunlight penetration and avoid overhead sprinkler irrigation."
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Lush cherry leaf displaying normal morphology and healthy chlorophyll pigmentation.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Ensure adequate winter dormancy chill hours and protect against bird damage."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "disease": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "is_healthy": False,
        "description": "Fungal pathogen causing narrow, rectangular, tan-to-gray lesions running parallel to leaf veins.",
        "treatment": "Apply strobilurin or triazole-based foliar fungicides (e.g., azoxystrobin) if infection starts before tasseling.",
        "prevention": "Practice 2-year crop rotation with non-host crops (like soybeans), deep till infected crop debris, and plant tolerant hybrids."
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "is_healthy": False,
        "description": "Fungal infection (Puccinia sorghi) causing small, powdery reddish-brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicides containing propiconazole or pyraclostrobin if rust appears on upper leaves before silking.",
        "prevention": "Plant rust-resistant hybrid seeds and plant early in the season to evade peak spore migration."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "is_healthy": False,
        "description": "Caused by Exserohilum turcicum, producing long, cigar-shaped grayish-green to tan lesions.",
        "treatment": "Spray foliar fungicides when lesions are visible on leaves below the ear leaf during tasseling.",
        "prevention": "Rotate crops with legumes, bury corn residues thoroughly after harvest, and sow resistant corn varieties."
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Vigorous corn leaf with continuous green lamina and strong parallel venation.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Maintain nitrogen and potassium balance and inspect regularly for armyworms or borers."
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "is_healthy": False,
        "description": "Guignardia bidwellii infection causing small reddish-brown circular spots with dark borders on leaves.",
        "treatment": "Apply myclobutanil or mancozeb fungicides from early shoot growth through fruit set.",
        "prevention": "Remove mummified berries from vines and ground, maintain open canopy training systems, and thin shoots."
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "is_healthy": False,
        "description": "Complex fungal wood disease causing 'tiger-stripe' interveinal chlorosis and necrosis.",
        "treatment": "No single chemical cure; seal large pruning wounds with pruning paint/wound sealant and cut back infected cordons.",
        "prevention": "Avoid pruning during damp rain conditions; disinfect pruning shears between each vine."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "is_healthy": False,
        "description": "Fungal leaf spot causing dark brown to black irregular necrotic spots surrounded by faint chlorotic halos.",
        "treatment": "Apply copper oxychloride, bordeaux mixture, or mancozeb at the first appearance of leaf spots.",
        "prevention": "Improve trellis airflow and ensure vineyard floor is clean of decaying leaf litter."
    },
    "Grape___healthy": {
        "crop": "Grape",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Uniform green grapevine foliage exhibiting healthy photosynthesis capacity.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Maintain optimal vine trellising and water management."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange (Citrus)",
        "disease": "Huanglongbing (Citrus Greening)",
        "is_healthy": False,
        "description": "Bacterial disease (Candidatus Liberibacter) spread by psyllid insects causing asymmetrical blotchy leaf mottle.",
        "treatment": "No cure once infected; inject zinc/micronutrient sprays to prolong life, but severely infected trees should be removed.",
        "prevention": "Aggressively manage Asian citrus psyllid vector using systemic imidacloprid and plant certified disease-free nursery stock."
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "description": "Xanthomonas arboricola causing water-soaked angular dark purple lesions that turn necrotic and drop out ('shot-hole').",
        "treatment": "Apply bactericidal fixed copper or oxytetracycline sprays starting at petal fall and shuck split.",
        "prevention": "Plant resistant cultivars, avoid high nitrogen fertilizers which promote overly soft succulent growth, and use windbreaks."
    },
    "Peach___healthy": {
        "crop": "Peach",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Healthy peach foliage with clean lanceolate leaves and clear margins.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Apply dormant copper spray in late winter to prevent peach leaf curl."
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "description": "Xanthomonas campestris producing small, water-soaked, blister-like spots that darken to brown/black.",
        "treatment": "Spray copper hydroxide combined with mancozeb weekly during warm wet periods.",
        "prevention": "Use certified pathogen-free seeds, practice strict drip irrigation (avoid wetting leaves), and rotate with non-solanaceous crops."
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Glossy green pepper leaf free of pustules, spots, or viral curling.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Provide consistent moisture and calcium to prevent blossom end rot on peppers."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "is_healthy": False,
        "description": "Alternaria solani fungal infection causing dark brown concentric rings ('target board' pattern) on older leaves.",
        "treatment": "Apply chlorothalonil, mancozeb, or azoxystrobin fungicides at 7-10 day intervals.",
        "prevention": "Destroy old potato haulms, rotate crops with brassicas/cereals, and avoid overhead sprinkler irrigation."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "is_healthy": False,
        "description": "Oomycete Phytophthora infestans causing rapidly expanding water-soaked dark lesions with white fungal fuzz underneath.",
        "treatment": "Immediate application of systemic fungicides like metalaxyl, cymoxanil, or copper fungicides.",
        "prevention": "Plant certified seed tubers, monitor local late-blight weather warnings, and hill potatoes well to shield tubers."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Vibrant green potato foliage without fungal spots or foliar blights.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Scout weekly for Colorado potato beetles and maintain good soil drainage."
    },
    "Raspberry___healthy": {
        "crop": "Raspberry",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Intact compound serrated raspberry leaves with vibrant vegetative health.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Prune out spent floricanes after summer fruiting to allow new primocanes room to grow."
    },
    "Soybean___healthy": {
        "crop": "Soybean",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Healthy trifoliate soybean leaf with intact mesophyll tissue.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Practice good crop rotation and control weed competition in early vegetative stages."
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "description": "Fungal growth (Podosphaera xanthii) coating leaf surfaces in talcum-powder-like white mycelium.",
        "treatment": "Spray potassium bicarbonate, sulfur, or horticultural oils (neem) at the very onset of white patches.",
        "prevention": "Space squash plants generously for ventilation and choose resistant cultivars."
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "disease": "Leaf Scorch",
        "is_healthy": False,
        "description": "Diplocarpon earlianum fungus producing dark purplish irregular blotches that scorch leaf edges.",
        "treatment": "Apply captan or thiophanate-methyl fungicides after post-harvest renovation.",
        "prevention": "Remove dead strawberry foliage, replace beds every 3-4 years, and irrigate via drip tapes."
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Healthy strawberry foliage with glossy deep green leaflets.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Mulch clean straw underneath plants to prevent soil splash on leaves and berries."
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "description": "Xanthomonas campestris pv. vesicatoria causing small (2-3mm) greasy, water-soaked brown specks on leaves.",
        "treatment": "Apply copper bactericide combined with mancozeb. Actigard (acibenzolar-S-methyl) can induce systemic resistance.",
        "prevention": "Disinfect garden stakes and tools, avoid overhead watering, and discard infected seedlings."
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "is_healthy": False,
        "description": "Fungal disease (Alternaria solani) producing characteristic concentric target-board dark brown lesions surrounded by yellow halos.",
        "treatment": "Apply copper fungicide, chlorothalonil, or mancozeb every 7 to 14 days. Prune off infected lower leaves immediately.",
        "prevention": "Mulch around base to prevent soil pathogens splashing onto foliage, rotate crops for 3 years, and stake plants for airflow."
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "is_healthy": False,
        "description": "Destructive water-mold (Phytophthora infestans) creating large, rapidly expanding water-soaked greasy gray/black patches.",
        "treatment": "Apply preventative biofungicide (Bacillus subtilis) or systemic fungicides (mefenoxam, chlorothalonil). Destroy severely sick plants immediately.",
        "prevention": "Plant resistant cultivars, destroy volunteer tomatoes/potatoes, and keep foliage dry."
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "is_healthy": False,
        "description": "Passalora fulva fungus producing pale green/yellow spots on top of leaves with olive-green velvety mold on undersides.",
        "treatment": "Apply copper fungicides or biofungicides; ensure high greenhouse exhaust ventilation.",
        "prevention": "Keep relative humidity below 85% in greenhouses and space plants at least 24 inches apart."
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "is_healthy": False,
        "description": "Septoria lycopersici fungus causing numerous circular spots with light gray centers and dark brown borders.",
        "treatment": "Spray copper fungicides or chlorothalonil at first symptom. Prune diseased lower foliage.",
        "prevention": "Clean up plant debris after harvest, apply clean straw mulch, and use soaker hoses rather than overhead sprinklers."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Spider Mites (Two-Spotted Spider Mite)",
        "is_healthy": False,
        "description": "Tetranychus urticae pest damage causing fine yellow stippling/bronzing and delicate webbing on leaf undersides.",
        "treatment": "Spray insecticidal soap, horticultural mineral oil, or neem oil. Introduce natural predators like Phytoseiulus persimilis mites.",
        "prevention": "Maintain adequate soil moisture (mites thrive in dry, dusty conditions) and wash foliage periodically with water jets."
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "is_healthy": False,
        "description": "Corynespora cassiicola fungus causing brown lesions with pale concentric rings and dark margins on foliage and stems.",
        "treatment": "Apply strobilurin fungicides (e.g., azoxystrobin) or copper formulations.",
        "prevention": "Provide wide plant spacing to reduce canopy humidity and avoid wetting leaves when watering."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "description": "Geminivirus transmitted by whiteflies (Bemisia tabaci) leading to severe stunting, upward leaf curling, and yellowing margins.",
        "treatment": "No cure for viral infections once established; rogue out and destroy infected plants immediately to protect neighbors.",
        "prevention": "Control whitefly vectors using yellow sticky traps and insect netting; choose certified TYLCV-resistant hybrids."
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "description": "Tobamovirus causing mottled light/dark green mosaic patterns, leaf distortion ('shoestringing'), and plant stunting.",
        "treatment": "No chemical treatment exists. Carefully pull up and burn/bag infected plants. Do not compost.",
        "prevention": "Wash hands with soap before touching plants (especially if using tobacco), sterilize tools in 10% bleach, and use resistant varieties."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "is_healthy": True,
        "description": "Healthy tomato leaf with lush green color and clean margins.",
        "treatment": "Healthy Leaf, no treatment needed.",
        "prevention": "Maintain regular watering schedule, balance N-P-K nutrition, and prune suckers to optimize yield."
    }
}


def get_disease_info(class_name: str) -> dict:
    """
    Safely retrieves crop, disease, status, treatment, and prevention info.
    If the class is unrecognized, returns a generic fallback.
    """
    if class_name in DISEASE_DETAILS:
        return DISEASE_DETAILS[class_name]

    # Heuristic parsing fallback for raw labels
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()

    return {
        "crop": crop,
        "disease": disease,
        "is_healthy": is_healthy,
        "description": f"Detected condition: {disease} on {crop}.",
        "treatment": "Healthy Leaf, no treatment needed." if is_healthy else "Consult local agricultural extension office for targeted fungicide/bactericide recommendations.",
        "prevention": "Maintain optimal cultural practices, sanitization, and moisture management."
    }
