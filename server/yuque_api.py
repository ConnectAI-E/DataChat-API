# """
# 获取  语雀的  body - 正文 Markdown 源代码
#
# 所有 API 的路径都以 https://www.yuque.com/api/v2 开头。
# 访问空间内资源的 API 需要使用空间对应的域名，例如空间域名为 customspace.yuque.com，
# 则 API 的基础路径为 https://customspace.yuque.com/api/v2。
#
# GET /repos/:namespace/docs/:slug
# 例如：https://www.yuque.com/yuque/developer/gyht993a76zg54mv
# namespace -》yuque/developer
# slug -》gyht993a76zg54mv
#
# """

import requests
from urllib.parse import urlparse

# 个人/团队token  ->  "X-Auth-Token": 'pL5V8y0Tq1bq1qYoNfaJnSmDAEIxne08MHarNaww'
# 在线文档链接  ->

def extract_paths(urls):
    a = []
    a.append(urls)
    urls = a
    paths, params = [], []
    for url in urls:
        parsed_url = urlparse(url)
        if parsed_url.path:
            path_segments = parsed_url.path.strip('/').split('/')
            paths.append(path_segments)
    paths = paths[0]
    if len(paths) > 2:
        namespace = paths[0] + '/' + paths[1]
        sulg = paths[2]
        params.append(namespace)
        params.append(sulg)
        return params
    elif len(paths) == 2:
        namespace = paths[0] + '/' + paths[1]
        params.append(namespace)
        return params


def request_data(params):
    if len(params) == 2:# 获取单个知识库的单个文档
        namespace, slug = params[0], params[1]
        url = "https://www.yuque.com/api/v2/repos/%s/docs/%s" % (namespace, slug)
        response = requests.request(method='GET', url=url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(data['data']['body'])
        else:
            print(f"请求失败，状态码: {response.status_code}")

    elif len(params) == 1: # 获取单个知识库的所有文档
        namespace = params[0]
        url = "https://www.yuque.com/api/v2/repos/%s/docs" % (namespace)
        response = requests.request(method='GET', url=url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(data['data']['body'])
        else:
            print(f"请求失败，状态码: {response.status_code}")


    # else: # 获取所有的知识库的列表
    #     namespace = params[0]
    #     path_segments = namespace[0].strip('/').split('/')
    #     url = "https://www.yuque.com/api/v2/users/%s/repos/" % (path_segments)
    #     response = requests.request(method='GET', url=url, headers=headers)
    #     if response.status_code == 200:
    #         data = response.json()
    #         print(data['data']['body'])
    #     else:
    #         print(f"请求失败，状态码: {response.status_code}")



if __name__ == "__main__":
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "yuque-hexo with python",
        "X-Auth-Token": 'pL5V8y0Tq1bq1qYoNfaJnSmDAEIxne08MHarNaww'
    }

    # urls = 'https://www.yuque.com/stopgivingafucks/lm15dv?# 《测试》'
    # urls = 'https://www.yuque.com/stopgivingafucks/lm15dv'
    # urls = 'https://www.yuque.com/stopgivingafucks/lm15dv/sumc11wq1txbfk0s'
    # urls = 'https://www.yuque.com/stopgivingafucks/lm15dv/rlrl9fo68dlb0x49'
    # urls = 'https://www.yuque.com/stopgivingafucks/lm15dv/zgc4zq42nhshc705'
    urls = 'https://www.yuque.com/stopgivingafucks/lm15dv/di8ng4qnxrcmqfph'

    # namespace, slug = "stopgivingafucks/lm15dv", "zwkh4sdwnibr19qq"

    params = extract_paths(urls)  # return  ['stopgivingafucks/lm15dv', 'zwkh4sdwnibr19qq'] 或者  ['stopgivingafucks/lm15dv']
    request_data(params)


