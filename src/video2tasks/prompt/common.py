"""Shared prompt blocks for cross-mode reuse."""

DEFAULT_HIGH_LEVEL_TASK = (
    "Put the toy cars into the brown basket and put the fruit into the black basket."
)

DEFAULT_OBJECT_INVENTORY_BLOCK = (
    "Object inventory (some props may repeat):\n"
    "- Containers:\n"
    "Descriptor : CanonicalRef\n"
    "A brown rectangular bamboo-woven storage basket : brown basket\n"
    "A black rectangular bamboo-woven storage basket : black basket\n"
    "- Fruit:\n"
    "A green round kiwifruit : kiwi\n"
    "Half a kiwifruit with a brown fuzzy skin and a green interior (a kind of ground fruit when look) : kiwi\n"
    "A purple bunch of grapes : grapes\n"
    "Half a red apple with a white interior : apple\n"
    "A yellow avocado with a white center and an orange pit : avocado\n"
    "A green avocado (alligator pear) with a bumpy, textured skin : avocado\n"
    "An orange with a bright orange peel : orange\n"
    "A small round orange-colored citrus fruit with a green stem : orange\n"
    "- Toy cars:\n"
    "A small red toy truck : red car\n"
    "A small red toy car : red car\n"
    "A small gold toy car : gold car\n"
    "A small black toy car : black car\n"
    "Various toy cars : <color> car\n"
    "Rule for Various toy cars: if the car color is identifiable, set CanonicalRef to \"<color> car\" (e.g., blue car, white car). Use a single color word.\n"
)
