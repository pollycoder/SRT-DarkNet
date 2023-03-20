import argparse
import traceback
import os
import sys
sys.path.append('..')
from helper import utils, log
sys.path.append('models')
from models.crawler import Crawler
import time 

if __name__ == '__main__':
    open_world = True
    # Init
    SOCKS_PORT = utils.USED_SOCKS_PORT
    CONTROLLER_PORT = utils.USED_CONTROL_PORT
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Crawl a list of URLs in several batches.')
    
    # Add arguments
    parser.add_argument('--ntabs', default=3, type=int, help='Number of tabs in the tor browser')
    parser.add_argument('--batch', default=5, type=int, help='Number of batches')
    parser.add_argument('--comb', default=4, type=int, help='Number of combinations')
    parser.add_argument('--instance', default=4, type=int, help='Number of instances')
    parser.add_argument('--output', default='output', type=str, help='Path of the output file')
    parser.add_argument('--tbbpath', default='../tbb/tor-browser_zh-CN', type=str, help='Path of tbb')  
    parser.add_argument('--xvfb', default=False, type=bool, help='Use XVFB (for headless testing)') 
    parser.add_argument('--screenshot', default=False, type=bool, help='Capture page screenshots)')
    parser.add_argument('--urls_check', default='', type=str, help='Alexa Top 1w URLs file path')
    
    args = parser.parse_args()

    # Load data
    num_tabs = args.ntabs
    num_batches = args.batch
    num_instances = args.instance
    num_comb = args.comb
    output = args.output
    tbb_path = args.tbbpath
    xvfb = args.xvfb
    screenshot = args.screenshot
    urls_check_path = args.urls_check

    urls_check_list = []
    assert os.path.isfile(urls_check_path)
    assert urls_check_path
    with open(urls_check_path, 'r') as fp:
        urls_check_list = fp.read().splitlines()

    
    torrc_dict = {'SocksPort': str(SOCKS_PORT),
             'ControlPort': str(CONTROLLER_PORT)}
    
    crawler = Crawler(torrc_dict, urls_check_list,open_world, num_tabs, tbb_path, output, xvfb, screenshot)
    print('INFO\tInit crawler finish in {}'.format(utils.cal_now_time()))
    print("INFO\tCommand line parameters: %s" % sys.argv)
    
    # Run the crawl
    
    try:
        crawler.crawl(num_batches, num_comb, num_instances)
    except KeyboardInterrupt:
        log.wl_log.warning("WARNING\tKeyboard interrupt! Quitting in {}...".format(utils.cal_now_time()))
    except Exception as e:
        log.wl_log.error("ERROR\tException in {}: \n {}".format(utils.cal_now_time(), traceback.format_exc()))
    finally:
        crawler.stop_crawl()
    
    print('INFO\tData Collection done in {}'.format(utils.cal_now_time()))