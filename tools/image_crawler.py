from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': 'great_horned_owl/'})
google_crawler.crawl(keyword='great horned owl', max_num=500,
                     date_min=None, date_max=None,
                     min_size=None, max_size=None)