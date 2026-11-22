#!/usr/bin/env python3
"""
Search the claims from the command line.

    python search_similar_claims.py --text "limping after a fall"
    python search_similar_claims.py --describe "a curly-haired dog"
    python search_similar_claims.py --xray data/PetXrays/<file>.png
    python search_similar_claims.py --photo data/PetPhotos/<file>.jpg
    python search_similar_claims.py --keywords "radiograph shoulder"
"""
import argparse

from claims import search, store


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="search the claim descriptions")
    group.add_argument("--describe", help="describe the pet: search the pet photos with words")
    group.add_argument("--xray", help="an X-ray image: claims with similar X-rays")
    group.add_argument("--photo", help="a pet photo: claims with similar pet photos")
    group.add_argument("--keywords", help="BM25 keyword search over the notes")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    client = store.connect()
    try:
        if args.text:
            results, what = search.by_text(client, args.text, args.limit), f'text "{args.text}"'
        elif args.describe:
            results, what = search.by_description(client, args.describe, args.limit), f'description "{args.describe}"'
        elif args.xray:
            results, what = search.by_xray(client, args.xray, args.limit), f"X-ray {args.xray}"
        elif args.photo:
            results, what = search.by_photo(client, args.photo, args.limit), f"photo {args.photo}"
        else:
            results, what = search.keyword(client, args.keywords, args.limit), f'keywords "{args.keywords}"'
    finally:
        client.close()
    print(f"Searching with {what}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {search.describe(result)}\n")


if __name__ == "__main__":
    main()
