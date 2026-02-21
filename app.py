from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = YOLO('best.pt')

# ─────────────────────────────────────────────
# FOLLOW-UP QUESTIONS
# Only triggered for ambiguous YOLO classes
# ─────────────────────────────────────────────
FOLLOWUP_QUESTIONS = {
    'paper': {
        'question': 'What type of paper item is this?',
        'options': [
            {'id': 'clean_paper',    'label': 'Clean / dry paper or newspaper'},
            {'id': 'cardboard',      'label': 'Clean cardboard box'},
            {'id': 'pizza_box',      'label': 'Pizza box or food-soiled cardboard'},
            {'id': 'juice_carton',   'label': 'Juice, milk, or drink carton'},
        ]
    },
    'plastic': {
        'question': 'What type of plastic item is this?',
        'options': [
            {'id': 'plastic_bottle', 'label': 'Bottle, jug, or tub (e.g. water bottle, shampoo)'},
            {'id': 'plastic_bag',    'label': 'Plastic bag or film'},
            {'id': 'plastic_utensil','label': 'Plastic utensil, straw, or cutlery'},
            {'id': 'plastic_toy',    'label': 'Plastic toy or non-bottle item'},
        ]
    },
    'wrapper': {
        'question': 'What type of wrapper is this?',
        'options': [
            {'id': 'food_wrapper',   'label': 'Food wrapper (chip bag, candy, snack)'},
            {'id': 'bubble_wrap',    'label': 'Bubble wrap or packing material'},
            {'id': 'plastic_film',   'label': 'Plastic film or shrink wrap'},
        ]
    },
}

# ─────────────────────────────────────────────
# CITY MAPPING TABLE
# Structure: city -> yolo_class -> subtype -> result
# Use subtype='default' for classes with no follow-up
# ─────────────────────────────────────────────
CITY_RULES = {
    'livermore': {
        'biowaste':   {'default': {'bin': 'Organic / Compost',  'color': 'green', 'emoji': '🌱', 'tip': 'Place in your green organics cart.'}},
        'glass':      {'default': {'bin': 'Recyclable',         'color': 'blue',  'emoji': '♻️', 'tip': 'Bottles and jars only. Rinse clean before recycling.'}},
        'metal':      {'default': {'bin': 'Recyclable',         'color': 'blue',  'emoji': '♻️', 'tip': 'Empty cans and tins. Rinse clean.'}},
        'thermocol':  {'default': {'bin': 'Trash / Garbage',    'color': 'black', 'emoji': '🗑️', 'tip': 'Styrofoam is NOT recyclable in Livermore. Goes in black trash cart.'}},
        'footware':   {'default': {'bin': 'Other',              'color': 'gray',  'emoji': '❓', 'tip': 'Contact Livermore Sanitation at 925-449-7300 for proper disposal options.'}},
        'cloth':      {'default': {'bin': 'Other',              'color': 'gray',  'emoji': '❓', 'tip': 'Donate if in good condition. Contact Livermore Sanitation at 925-449-7300 for textile disposal.'}},
        'paper': {
            'clean_paper':     {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Place in blue recycling cart. Must be dry and clean.'},
            'cardboard':       {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Flatten boxes and place in blue recycling cart.'},
            'pizza_box':       {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Livermore accepts pizza boxes in recycling unless heavily soiled with grease. If very greasy, place in trash.'},
            'juice_carton':    {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Rinse carton and place in blue recycling cart.'},
        },
        'plastic': {
            'plastic_bottle':  {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Bottles, jugs and tubs #1-#7. Rinse and place in blue recycling cart.'},
            'plastic_bag':     {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Plastic bags are NOT accepted in Livermore recycling. Place in trash.'},
            'plastic_utensil': {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Single-use utensils go in the trash in Livermore.'},
            'plastic_toy':     {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Broken plastic toys go in trash. Donate if in good condition.'},
        },
        'wrapper': {
            'food_wrapper':    {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Chip bags and food wrappers go in the black trash cart.'},
            'bubble_wrap':     {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Bubble wrap and plastic film are not accepted curbside. Place in trash.'},
            'plastic_film':    {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Plastic film goes in trash. Some grocery stores have drop-off bins for film recycling.'},
        },
    },

    'oakland': {
        'biowaste':   {'default': {'bin': 'Organic / Compost',  'color': 'green', 'emoji': '🌱', 'tip': 'Place in your green compost cart. Do NOT use plastic bags.'}},
        'glass':      {'default': {'bin': 'Recyclable',         'color': 'blue',  'emoji': '♻️', 'tip': 'Bottles and jars only. Empty and rinse clean.'}},
        'metal':      {'default': {'bin': 'Recyclable',         'color': 'blue',  'emoji': '♻️', 'tip': 'Empty cans and tins go in the blue recycling cart.'}},
        'thermocol':  {'default': {'bin': 'Trash / Garbage',    'color': 'black', 'emoji': '🗑️', 'tip': 'Polystyrene/Styrofoam is NOT recyclable in Oakland. Goes in black trash cart.'}},
        'footware':   {'default': {'bin': 'Other',              'color': 'gray',  'emoji': '❓', 'tip': 'Contact Oakland Recycles or schedule a bulky item pickup at oaklandrecycles.com.'}},
        'cloth':      {'default': {'bin': 'Other',              'color': 'gray',  'emoji': '❓', 'tip': 'Textiles do NOT go in Oakland recycling. Donate if usable or contact Oakland Recycles for textile drop-off options.'}},
        'paper': {
            'clean_paper':     {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Clean dry paper goes in blue recycling cart.'},
            'cardboard':       {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Flatten boxes. Place in blue recycling cart.'},
            'pizza_box':       {'bin': 'Organic / Compost',   'color': 'green', 'emoji': '🌱', 'tip': 'Oakland: food-soiled pizza boxes go in the GREEN compost cart, not recycling.'},
            'juice_carton':    {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Oakland does not accept juice/milk cartons curbside. Place in trash.'},
        },
        'plastic': {
            'plastic_bottle':  {'bin': 'Recyclable',          'color': 'blue',  'emoji': '♻️', 'tip': 'Plastic bottles, jugs, tubs resins #1, #2, #5 only. Rinse clean.'},
            'plastic_bag':     {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Plastic bags are NOT accepted in Oakland recycling. Place in trash.'},
            'plastic_utensil': {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Single-use plastic utensils go in the trash in Oakland.'},
            'plastic_toy':     {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Broken plastic toys go in trash. Donate if in good condition.'},
        },
        'wrapper': {
            'food_wrapper':    {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Chip bags, candy wrappers go in black trash cart in Oakland.'},
            'bubble_wrap':     {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Bubble wrap is not accepted curbside in Oakland. Place in trash.'},
            'plastic_film':    {'bin': 'Trash / Garbage',     'color': 'black', 'emoji': '🗑️', 'tip': 'Plastic film goes in trash. Some stores have drop-off bins.'},
        },
    }
}

CITY_CONTACT = {
    'livermore': 'Livermore Sanitation: 925-449-7300 | livermorerecycles.org',
    'oakland':   'Oakland Recycles: oaklandrecycles.com | 311 for service issues'
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    city = request.form.get('city', '').lower()
    if city not in CITY_RULES:
        return jsonify({'error': 'Invalid city. Choose livermore or oakland.'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_w, img_h = img.size

    results = model(img, conf=0.6, max_det=5)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            bbox = box.xyxyn[0].tolist()  # normalized [x1,y1,x2,y2]

            city_class_rules = CITY_RULES[city].get(cls_name, {})
            needs_followup = cls_name in FOLLOWUP_QUESTIONS and 'default' not in city_class_rules

            detection = {
                'id': f"{cls_name}_{len(detections)}",
                'label': cls_name,
                'confidence': round(confidence * 100, 1),
                'bbox': bbox,
                'needs_followup': needs_followup,
                'followup': FOLLOWUP_QUESTIONS.get(cls_name) if needs_followup else None,
                'result': None  # filled in if no followup needed
            }

            if not needs_followup:
                rule = city_class_rules.get('default')
                if rule:
                    detection['result'] = {
                        'bin': rule['bin'],
                        'color': rule['color'],
                        'emoji': rule['emoji'],
                        'tip': rule['tip'],
                        'city_contact': CITY_CONTACT[city]
                    }

            detections.append(detection)

    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return jsonify({'detections': detections, 'city': city})


@app.route('/resolve', methods=['POST'])
def resolve():
    """
    Called when user answers a follow-up question.
    Body: { city, yolo_class, subtype }
    Returns the bin recommendation for that subtype + city.
    """
    data = request.get_json()
    city = data.get('city', '').lower()
    yolo_class = data.get('yolo_class', '')
    subtype = data.get('subtype', '')

    if city not in CITY_RULES:
        return jsonify({'error': 'Invalid city'}), 400

    rule = CITY_RULES[city].get(yolo_class, {}).get(subtype)
    if not rule:
        return jsonify({'error': 'No rule found for this combination'}), 404

    return jsonify({
        'bin': rule['bin'],
        'color': rule['color'],
        'emoji': rule['emoji'],
        'tip': rule['tip'],
        'city_contact': CITY_CONTACT[city]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
