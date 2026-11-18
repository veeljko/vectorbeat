"""Share of downloaded pages that are relevant to the crawl topic."""

from collections.abc import Collection, Iterable


def harvest_rate(links: Collection[str], relevant_urls: Iterable[str]) -> float:
    """Return the fraction of links found in the relevant-URL collection.

    relevant_urls contains URLs classified as on-topic, such as pages whose
    content score reaches the crawl threshold. An empty collection returns 0.
    """
    if not links:
        return 0.0
    relevant_urls = set(relevant_urls)
    on_topic = 0
    for url in links:
        if url in relevant_urls:
            on_topic += 1
    return on_topic / len(links)
