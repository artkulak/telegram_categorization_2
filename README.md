# Data Clustering Contest 2021. Round 2
The task in the second round is to improve the C/C++ library you created in the [previous round](https://contest.com/docs/dc2021-r1) that can determine the topic of a Telegram channel.
**Team: Tanned Gull**


* Team Members:
	- https://github.com/artkulak
	- https://github.com/iamAzeem
  - https://github.com/bezbahen0

## The task
Improve topic detection. In this round, you will need to improve topic detection for channels in English and Russian and determine the topic of channels in three additional languages. Required languages:

- English
- Russian
- Arabic
- Persian
- Uzbek

### List of Categories
Below is the full list of possible topics for this round. The list also includes some of the sorting recommendations used by human moderators to sort the evaluation datasets.

  - Art & Design
  - Bets & Gambling – includes sports bets
  - Books
  - Business & Entrepreneurship
  - Cars & Other Vehicles
  - Celebrities & Lifestyle
  - Cryptocurrencies
  - Culture & Events
  - Curious Facts
  - Directories of Channels & Bots
  - Drug Sale
  - Economy & Finance
  - Education
  - Erotic Content
  - Fashion & Beauty
  - Fitness
  - Forgery – includes fake documents, fake money, etc.
  - Food & Cooking
  - Foreign Language Learning
  - Hacked Accounts & Software – includes carding, passwords for subscription services, etc.
  - Health & Medicine
  - History
  - Hobbies & Activities
  - Home & Architecture
  - Humor & Memes
  - Investments
  - Job Listings
  - Kids & Parenting
  - Marketing & PR
  - Motivation & Self-development - includes inspirational quotes and poetry
  - Movies
  - Music
  - Offers & Promotions – includes products or services for sale, unless they fall under the newly added categories
  - Personal Data – includes doxxing, databases
  - Pets
  - Pirated Content – films, music, books, but not software
  - Politics & Incidents
  - Prostitution
  - Psychology & Relationships
  - Real Estate
  - Recreation & Entertainment
  - Religion & Spirituality
  - Science
  - Spam & Fake Followers – includes spam tools and services, boosting followers, likes, etc.
  - Sports – includes e-sports
  - Technology & Internet
  - Travel & Tourism
  - Video Games
  - Weapon Sale
  - Other

## Code Structure

```text
telegram_categorization_2
- notebooks - (contains training jupyter notebook for the contest)
- resources
  - fasttext            (dependency)
  - Tokenizer           (dependency)
  - LightGBM            (dependency)
  - libtgcat            (existing code)
  - libtgcat-tester     (existing code - used for testing)
  - models              (make sure that the model you use is in this directory)
- src
  - libtgcat            (updated libtgcat source - to be tested with libtgcat-tester-r2)
    - text_lightgbm     (lightgbm lightgbm adapted for text classification problem)
```

Make sure you download and place `lid.176.bin` in the `resources/models` directory.  
The `models/language` is a symlink to `lid.176.bin` and is used inside `libtgcat` to
load the language model for categorization. You can create your own symlink `language`
under `models` directory to any arbitrary location. This removes the need to compile 
the library everytime a different model is used.

Also, you can put your test files in `test-data` directory and use its symlink in the
`libtgcat-tester-r2/build` for testing purposes.


## Models

To get the models, you need to run notebooks/fasttext-text-classification.ipynb, the only model that is not there is the fasttext language model, it can be downloaded [here](https://fasttext.cc/docs/en/language-identification.html)

## Build

1. Build [fasttext](./resources/fastText/) library:

```shell
cd resources/fastText
mkdir build && cd build && cmake ..
make
```
The library `libfasttext.so` will be generated in the `build` folder.

2. Build [LightGBM](./resources/LightGBM/) library:

```shell
cd resources/LightGBM
mkdir build && cd build && cmake ..
make
```
The library `lib_lightgbm.so` will be generated in the `./resources/LightGBM` folder.

3. Build [Tokenizer](./resources/Tokenizer/) library:

```shell
cd resources/Tokenizer
mkdir build && cd build && cmake ..
make
```
The library `libOpenNMTTokenizer.so` will be generated in the `build` folder.


4. Build [src/libtgcat](./src/libtgcat/) - the updated library:

```shell
cd src/libtgcat
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

The `libtgcat` library loads the model i.e. `./resources/models/lid.176.bin`
and is linked with `/resources/fasttext/build/libfasttext.so`.

5. Build [resources/libtgcat-tester-r2](./resources/libtgcat-tester-r2/):

Make sure that the `libtgcat` is already built and its `libtgcat.so` file is
located in its `build` folder.

```shell
cd resources/libtgcat-tester-r2
ln -s ../../src/libtgcat/build/libtgcat.so .
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Test

Make sure `libtgcat-tester-r2` has been built and its executable is under its
`build` folder.

```shell
cd resources/libtgcat-tester-r2/build
```

Using `d1k.txt` file from `test-data` folder:

```shell
$ ./tgcat-tester language ./../../test-data/d1k.txt ./o1k.txt
Processed 1000 queries in 0.699325 seconds
```

You can create symlink of `test-data` in this directory for a better workflow:

```shell
$ ln -s ./../../test-data/ .
$ ./tgcat-tester language ./test-data/d1k.txt ./test-data/o1k.txt
Processed 1000 queries in 0.699325 seconds
```

Now, all the inputs and output will be in the `test-data` folder.
