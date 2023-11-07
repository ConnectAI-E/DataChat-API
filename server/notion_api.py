import requests
import json
import asyncio
import re


# 用来对urls地址中的id号进行提取处理
def extract_ids(link):
    urls = []
    urls.append(link)
    ids, id = [], []
    pattern = r'https://www.notion.so/([-a-f0-9]+)'
    for url in urls:
        match = re.search(pattern, url)
        if match:
            ids.append(match.group(1))
    # 获取后32位
    for i in ids:
        id.append(i[-32:])
        id = id[0]
    id = id[0:8] + '-' + id[8:12] + '-' + id[12:16] + '-' + id[16:20] + '-' + id[20:]
    return id


# 获取到  block  中  paragraph.rich_text[*].plain_text  的富文本内容
def get_plain_text_from_rich_text(rich_text):
    return "".join([t['plain_text'] for t in rich_text])


def get_media_source_text(block):

    if (block[block.type].external):
        source = block[block.type].external.url
    elif (block[block.type].file):
        source = block[block.type].file.url
    elif (block[block.type].url):
        source = block[block.type].url
    else:
        source = "[缺少媒体块类型的大小写]: " + block.type

    if (block[block.type].caption.length):
        caption = get_plain_text_from_rich_text(block[block.type].caption)
        return caption + ": " + source

    return source


def get_text_from_block(block):
    text = ""
    # 这里加异常处理原因是，获取到的content中不一定有 block[block['type']]['rich_text']
    try:
        if block[block['type']]['rich_text']:
            text = get_plain_text_from_rich_text(block[block['type']]['rich_text'])
    except:
        block_type = block['type']
        if block_type == "unsupported":
            text = "[Unsupported block type]"

        elif block_type == "bookmark":
            text = block['bookmark']['url']

        elif block_type == "child_database":
            text = block['child_database']['title']

        elif block_type == "child_page":
            text = block['child_page']['title']

        elif block_type in ["embed", "video", "file", "image", "pdf"]:
            text = get_media_source_text(block)

        elif block_type == "equation":
            text = block['equation']['expression']

        elif block_type == "link_preview":
            text = block['link_preview']['url']

        elif block_type == "synced_block":
            if 'synced_from' in block['synced_block']:
                synced_with_block = block['synced_block']['synced_from']
                text = f"This block is synced with a block with the following ID: {synced_with_block[synced_with_block['type']]}"
            else:
                text = "Source sync block that another blocked is synced with."

        elif block_type == "table":
            text = f"Table width: {block['table']['table_width']}"

        elif block_type == "table_of_contents":
            text = f"ToC color: {block['table_of_contents']['color']}"

        elif block_type in ["breadcrumb", "column_list", "divider"]:
            text = "No text available"

        else:
            text = "[Needs case added]"

    string_data = f"{block['type']}: {text}"
    # 使用冒号分割字符串
    key, value = string_data.split(':')
    # 创建字典
    my_dict = {key.strip(): value.strip()}

    # 只返回正文的内容
    if 'paragraph' in my_dict and my_dict['paragraph'] != '':
        return my_dict['paragraph']
    else:
        return ''

    # return f"{block['type']}: {text}"


async def retrieve_block_children(page_id):

    """
    这里的  url  是接受处理过的id号拼接为的接口地址
    """
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"


    headers = {
        "Authorization": Authorization,
        "Notion-Version": Notion_Version
    }

    blocks = []
    cursor = None

    while True:
        params = {}
        if cursor:
            params["start_cursor"] = cursor

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Failed to retrieve blocks: {response.status_code}")
            return blocks

        data = response.json()
        blocks.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        if not has_more:
            break
        cursor = data.get("next_cursor")

    return blocks


# 输出获取到的blocks中的内容
def print_block_text(blocks):
    # 一个id可能会对应多个blocks，需要遍历的方式获取到每个blocks的内容
    for block in blocks:
        # 根据相应的规则，拿到内容
        text = get_text_from_block(block)
        print(text)


async def main(page_id):
    blocks = await retrieve_block_children(page_id)
    print_block_text(blocks)


if __name__ == "__main__":

    Notion_Version = '2022-06-28'
    Authorization = 'Bearer secret_j8nkz86I0vFVoVKN107SEQFKOW1MtSe9DflGfYh4w9L'

    """
    测试链接：
    https://www.notion.so/b1-8beaa48d081e44e69000cd789726a151
    https://www.notion.so/b1-8beaa48d081e44e69000cd789726a151?pvs=4
    https://www.notion.so/12312312-9eca1eb63e844d799a1ad171c9fd36a3
    """
    # page_id = extract_ids('https://www.notion.so/b1-8beaa48d081e44e69000cd789726a151?pvs=4')
    page_id = extract_ids('https://www.notion.so/b1-8beaa48d081e44e69000cd789726a151')
    # page_id = extract_ids('https://www.notion.so/12312312-9eca1eb63e844d799a1ad171c9fd36a3')


    asyncio.run(main(page_id))
