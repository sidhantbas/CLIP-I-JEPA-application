"""darkroom: where photographs are developed into sentences.

A miniature ClipCap. A frozen CLIP eye embeds the photograph, a frozen
GPT-2 tongue speaks, and the only part that trains is the developer: a
small mapping network that turns one CLIP embedding into a handful of
prefix tokens GPT-2 can continue. Flickr8k is the darkroom's contact
sheet; everything trains on an Apple M4 Pro in minutes.
"""
