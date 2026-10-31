# TrashSort

Take a photo of something you're throwing away and the app tells you which bin it goes in: recycling, compost, trash, or a special drop-off (batteries, clothes, shoes).

It uses an image model we trained ourselves, and the model runs right on the phone. There are no accounts, no API keys, and no cost per photo, and photos never leave the phone.

## Try it

```bash
npm install
npx expo start
```

Press `w` to open it in your browser. To try it on your phone, connect it to the same Wi-Fi as your computer and open the `http://192.168...:8081` address that `npx expo start` prints, in the phone's browser.

The model only runs in the web version for now. The Expo Go app can't load it, so use the phone's browser.

## Share it with your school

The app builds into a plain website, a folder of files you can put on any web host:

```bash
npx expo export --platform web
```

That makes a `dist/` folder. Some free ways to put it online:

- **EAS Hosting:** `npx eas-cli@latest login`, then `npx eas-cli@latest deploy --prod`. You get a link like `https://trashsort.expo.app`.
- **Netlify Drop:** drag the `dist/` folder onto https://app.netlify.com/drop.
- Your school's own website, if IT can upload a folder.

Students open the link on their phones and tap "Take a photo". They can use "Add to Home Screen" so it opens like an app. The first visit downloads about 20 MB (the model and the code that runs it), and after that the browser keeps it.

## What it knows

| Looks like | Bin |
| --- | --- |
| Cardboard, paper, glass bottles and jars, metal cans, plastic bottles and containers | Recycling |
| Food scraps, paper towels and napkins | Compost |
| Plastic bags and wrappers, styrofoam, takeaway cups, disposable plates, general trash | Trash |
| Batteries, clothes, shoes | Special drop-off |

It works best with one item, up close. It only knows these 15 kinds of things, so for anything else (a phone, a toy) it picks whichever of the 15 looks closest. Recycling rules also differ between cities. To change which bin something goes in, edit `src/lib/items.ts` and `ITEM_BINS` in `training/train.py`.

## How well it works

We tested it two ways (run `training/evaluate.py` to repeat this):

- **Photos like the ones it learned from:** 1,961 photos held back from training. It picks the right bin **98.5%** of the time.
- **Photos taken by someone else, in a different style:** [TrashNet](https://huggingface.co/datasets/garythung/trashnet), 2,527 photos of trash on a white background, never used for training. It picks the right bin **78.6%** of the time. It's good with cardboard and glass (about 90%), OK with plastic (83%), weaker with metal and paper (about 73%), and usually wrong on general trash (26%).

The second number is closer to what to expect from real phone photos. When the app adds "Not totally sure", take that seriously: on TrashNet it was right 87% of the time when it sounded sure, but only 56% of the time when it said it wasn't. Styrofoam is the weakest item, because there were only a few dozen good styrofoam photos to learn from.

Your own photos (next section) are the best way to make it better.

## Make it smarter with your own photos

Most of the training photos come from the internet, often on clean backgrounds. Real photos of real trash from your school will help the model the most.

1. Take photos of trash, one item per photo, saved as JPEG or PNG.
2. Put them in folders named after the item type: `training/data/school/plastic_film/`, `training/data/school/food/`, and so on. The folder names are the item types in the table above, as spelled in `src/lib/items.ts`.
3. Retrain (below). Your photos get mixed in with the rest, and some are held back to test with.

## Retraining the model

You need Python and [uv](https://docs.astral.sh/uv/). It takes about 20 minutes to download the photos and 15 to train on an M-series Mac.

```bash
cd training
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
.venv/bin/python download.py   # about 24,000 photos
.venv/bin/python clean.py      # removes mislabeled photos
.venv/bin/python train.py      # trains, tests, and saves the model into the app
.venv/bin/python evaluate.py   # tests the model that is in the app
```

`train.py` writes `public/model/trashsort.onnx` and `src/lib/model-labels.json`, so the app uses the new model on its next build.

### Where the photos come from

- [omasteam/waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) (MIT): the main set, about 20,000 photos in 10 groups.
- [viola77data/recycling-dataset](https://huggingface.co/datasets/viola77data/recycling-dataset) (Apache 2.0): plates, paper towels, takeaway cups, styrofoam, and more. It was collected by an automatic image search, so `clean.py` filters out photos that don't show what their folder says.
- ImageNet's plastic bag photos, via [mlnomad/imnet1k_plastic_bag](https://huggingface.co/datasets/mlnomad/imnet1k_plastic_bag). ImageNet is for non-commercial research and education.
- [TrashNet](https://huggingface.co/datasets/garythung/trashnet) (MIT): used only for testing, never for training.

The model is MobileNetV3, a small image model designed for phones, which starts out already knowing about everyday objects. `train.py` teaches it these 15 item types. In the browser, [ONNX Runtime Web](https://onnxruntime.ai/) runs it.
