export type Bin = 'recycling' | 'compost' | 'trash' | 'special';

// Every item type the model can recognize. The bins must match ITEM_BINS in training/train.py.
export const ITEMS = {
  battery: {
    name: 'Battery',
    bin: 'special',
    tip: 'Never put batteries in a bin: they can start fires in garbage trucks. Take them to a battery drop-off (lots of stores have one).',
  },
  cardboard: {
    name: 'Cardboard',
    bin: 'recycling',
    tip: 'Flatten it first. If it’s greasy, like a pizza box, it goes in compost instead.',
  },
  clothes: {
    name: 'Clothes',
    bin: 'special',
    tip: 'Donate them if they can still be worn, or take them to a clothing drop-off.',
  },
  disposable_plate: {
    name: 'Disposable plate',
    bin: 'trash',
    tip: 'Plain paper plates can go in compost. Plastic or shiny coated ones go in trash.',
  },
  food: {
    name: 'Food scraps',
    bin: 'compost',
    tip: 'Take off any stickers or packaging first.',
  },
  glass: {
    name: 'Glass bottle or jar',
    bin: 'recycling',
    tip: 'Rinse it out. Broken glass and drinking glasses go in trash.',
  },
  metal: {
    name: 'Metal can',
    bin: 'recycling',
    tip: 'Rinse it out first.',
  },
  paper: {
    name: 'Paper',
    bin: 'recycling',
    tip: 'Keep it dry. Used tissues and napkins go in compost.',
  },
  paper_towel: {
    name: 'Paper towel or napkin',
    bin: 'compost',
    tip: '',
  },
  plastic: {
    name: 'Plastic bottle or container',
    bin: 'recycling',
    tip: 'Empty and rinse it first.',
  },
  plastic_film: {
    name: 'Plastic bag or wrapper',
    bin: 'trash',
    tip: 'Don’t put bags in recycling: they jam the sorting machines. Many grocery stores collect clean plastic bags.',
  },
  shoes: {
    name: 'Shoes',
    bin: 'special',
    tip: 'Donate them if they can still be worn.',
  },
  styrofoam: {
    name: 'Styrofoam',
    bin: 'trash',
    tip: 'Most places can’t recycle foam, even when it has a recycling symbol.',
  },
  takeaway_cup: {
    name: 'Takeaway cup',
    bin: 'trash',
    tip: 'Paper coffee cups have a plastic lining, so most places can’t recycle them.',
  },
  trash: {
    name: 'General trash',
    bin: 'trash',
    tip: '',
  },
} satisfies Record<string, { name: string; bin: Bin; tip: string }>;

export type ItemId = keyof typeof ITEMS;

export type Prediction = {
  item: ItemId;
  bin: Bin;
  // How much of the model's belief went to this bin, from 0 to 1.
  confidence: number;
};
