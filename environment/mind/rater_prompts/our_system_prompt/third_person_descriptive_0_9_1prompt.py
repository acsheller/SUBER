import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive_0_9 import ThirdPersonDescriptive09_OurSys

from environment.memory import UserNewsInteraction
from environment.mind import News, NewsLoader
from environment.users import User


class ThirdPersonDescriptive09_1Shot_OurSys(ThirdPersonDescriptive09_OurSys):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_user=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None
        self.switch_user = switch_user
        print("---- 0_9_1 ----")

    def _get_few_shot_prompts(self):
        items_loader = NewsLoader()

        if self.cache_few_shot_prompts is None:
            if self.switch_user:

                user = User(
                    "Martin",
                    "M",
                    53,
                    (

                        "Martin Donovan, a seasoned bricklayer, has a keen interest in staying"
                        " updated with the latest news, especially related to the insurance sector."
                        " He believes that staying educated about the changes and trends in the insurance"
                        " industry helps him make informed decisions when it comes to his personal and"
                        " professional insurances.\n"

                        "He begins his day by reading the paper with a strong cup of coffee. Throughout the" 
                        " day, he takes breaks to scroll through various news apps on his phone, always keeping"
                        " an eye out for articles and updates about insurance policies, new insurance companies,"
                        " and trends in the industry.\n" 

                        "Martin also has a secondary interest in news related to social media. He loves learning"
                        " about the latest social media trends, updates, and controversies. He believes this"
                        " knowledge helps him connect better with his children and understand the world they are"
                        "growing up in."
                    ),
                )

                ####
                news_articles = items_loader.load_items_from_ids(['N55528', 'N16016', 'N61837', 'N53526'])
                news_article = news_articles[0]
                num_interacted = 0
                interactions = [
                UserNewsInteraction(3, 0, 1),
                UserNewsInteraction(4, 0, 1),
                UserNewsInteraction(3, 0, 1),
            ]
            retrieved_items = news_articles[1:]
            prompt = self._get_prompt(
                user,
                news_article,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation = (

                "Rating: 2 on a scale of 0 to 9, because Martin’s primary interest is"
                " in insurance news, with a secondary interest in social media. The article’s"
                " focus on consumer goods and branding related to the Royal Family does not align with"
                " these interests. While the lifestyle aspect of the story might appeal to him slightly"
                " due to its potential presence on social media, it lacks the direct relevance to insurance"
                " or social media trends that he usually seeks in news articles, resulting in a lower rating."

            )
        else:
            user = User(
                "Alex",
                "M",
                25,
                (
                    "Alex is a voracious consumer of news, particularly in the realm"
                    " of law and justice. As a journalist, he feels a responsibility to"
                    " stay current on the latest developments in legal matters, court"
                    " decisions, and societal implications of justice-related issues."
                    " He subscribes to various law journals and frequently visits websites"
                    " like The New York Times, BBC News, and The Guardian for their coverage"
                    " of legal affairs.\n"

                    "In addition to his primary interest, Alex follows football news closely. Being"
                    " a lifelong fan of the sport, he enjoys keeping up with his favorite teams and"
                    " players, as well as the general state of the sport. He checks ESPN and other"
                    " sports news websites daily and often watches live matches during his free time.\n"

                    "Alex follows the news not only for his job but also due to a deep-rooted interest in"
                    " current affairs. He believes that staying informed about the world helps him to be a"
                    " responsible citizen and to understand the context of his own life better. He often engages"
                    " in discussions about news stories on social media, participates in local town hall meetings,"
                    " and occasionally writes opinion pieces for his local newspaper."                    
                ),
            )
            news_articles = items_loader.load_items_from_ids(['N55528', 'N19639', 'N61837', 'N53526'])
            news_article = news_articles[0]
            num_interacted = 0
            interactions = [
            UserNewsInteraction(4, 0, 1),
            UserNewsInteraction(7, 0, 1),
            UserNewsInteraction(4, 0, 1),
            ]

            retrieved_items = news_articles[1:]
            prompt = self._get_prompt(
                user,
                news_article,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )

            explanation = (
                "Given Alex's reading habits and previous ratings, it is likely that"
                " he would rate the article 'The Brands Queen Elizabeth, Prince Charles, and" 
                " Prince Philip Swear By' a 3 on a scale of 0 to 9. Despite his primary interest in"
                " legal and justice-related news, Alex has demonstrated an appreciation for current affairs"
                " and lifestyle topics, albeit to a lesser extent. The article on the Royal Family's preferred"
                " brands offers an interesting glimpse into their personal preferences, which might capture" 
                " Alex’s curiosity to some degree. However, his focus on more impactful news, such as legal" 
                " matters and football, means that this lifestyle article would only moderately engage"
                " him, leading him to rate it a 3."
            )

            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {
                    "role": "assistant",
                    "content": prompt[1]["content"] + "\n\n" + explanation,
                },
            ]
        return self.cache_few_shot_prompts



class ThirdPersonDescriptive09_1Shot_OurSys_large(ThirdPersonDescriptive09_OurSys):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_user=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None
        self.switch_user = switch_user
        print("---- 0_9_1 ----")

    def _get_few_shot_prompts(self):
        items_loader = NewsLoader()

        if self.cache_few_shot_prompts is None:
            if self.switch_user:

                user = User(
                    "Martin",
                    "M",
                    53,
                    (

                        "Martin Donovan, a seasoned bricklayer, has a keen interest in staying"
                        " updated with the latest news, especially related to the insurance sector."
                        " He believes that staying educated about the changes and trends in the insurance"
                        " industry helps him make informed decisions when it comes to his personal and"
                        " professional insurances.\n"

                        "He begins his day by reading the paper with a strong cup of coffee. Throughout the" 
                        " day, he takes breaks to scroll through various news apps on his phone, always keeping"
                        " an eye out for articles and updates about insurance policies, new insurance companies,"
                        " and trends in the industry.\n" 

                        "Martin also has a secondary interest in news related to social media. He loves learning"
                        " about the latest social media trends, updates, and controversies. He believes this"
                        " knowledge helps him connect better with his children and understand the world they are"
                        "growing up in."
                    ),
                )


                news_articles = items_loader.load_items_from_ids(['N37683', 'N8006', 'N54264', 'N53526'])
                news_article = news_articles[0]
                num_interacted = 0
                interactions = [
                UserNewsInteraction(3, 0, 1),
                UserNewsInteraction(4, 0, 1),
                UserNewsInteraction(3, 0, 1),
            ]
                retrieved_items = news_articles[1:]
                prompt = self._get_prompt(
                    user,
                    news_article,
                    num_interacted,
                    interactions,
                    retrieved_items,
                    do_rename=False,
                )
                explanation = (

                    "Rating: 2 on a scale of 0 to 9, because Martin’s primary interest is"
                    " in insurance news, with a secondary interest in social media. The article’s"
                    " focus on consumer goods and branding related to the Royal Family does not align with"
                    " these interests. While the lifestyle aspect of the story might appeal to him slightly"
                    " due to its potential presence on social media, it lacks the direct relevance to insurance"
                    " or social media trends that he usually seeks in news articles, resulting in a lower rating."

                )
            else:
                user = User(
                    "Alex",
                    "M",
                    25,
                    (
                        "Alex is a voracious consumer of news, particularly in the realm"
                        " of law and justice. As a journalist, he feels a responsibility to"
                        " stay current on the latest developments in legal matters, court"
                        " decisions, and societal implications of justice-related issues."
                        " He subscribes to various law journals and frequently visits websites"
                        " like The New York Times, BBC News, and The Guardian for their coverage"
                        " of legal affairs.\n"

                        "In addition to his primary interest, Alex follows football news closely. Being"
                        " a lifelong fan of the sport, he enjoys keeping up with his favorite teams and"
                        " players, as well as the general state of the sport. He checks ESPN and other"
                        " sports news websites daily and often watches live matches during his free time.\n"

                        "Alex follows the news not only for his job but also due to a deep-rooted interest in"
                        " current affairs. He believes that staying informed about the world helps him to be a"
                        " responsible citizen and to understand the context of his own life better. He often engages"
                        " in discussions about news stories on social media, participates in local town hall meetings,"
                        " and occasionally writes opinion pieces for his local newspaper."                    
                    ),
                )
                news_articles = items_loader.load_items_from_ids(['N37683', 'N8006', 'N54264', 'N53526'])
                news_article = news_articles[0]
                num_interacted = 0
                interactions = [
                UserNewsInteraction(4, 0, 1),
                UserNewsInteraction(7, 0, 1),
                UserNewsInteraction(4, 0, 1),
                ]

                retrieved_items = news_articles[1:]
                prompt = self._get_prompt(
                    user,
                    news_article,
                    num_interacted,
                    interactions,
                    retrieved_items,
                    do_rename=False,
                )

                explanation = (
                    "Given Alex's reading habits and previous ratings, it is likely that"
                    " he would rate the article 'The Brands Queen Elizabeth, Prince Charles, and" 
                    " Prince Philip Swear By' a 3 on a scale of 0 to 9. Despite his primary interest in"
                    " legal and justice-related news, Alex has demonstrated an appreciation for current affairs"
                    " and lifestyle topics, albeit to a lesser extent. The article on the Royal Family's preferred"
                    " brands offers an interesting glimpse into their personal preferences, which might capture" 
                    " Alex’s curiosity to some degree. However, his focus on more impactful news, such as legal" 
                    " matters and football, means that this lifestyle article would only moderately engage"
                    " him, leading him to rate it a 3."
                )

            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {
                    "role": "assistant",
                    "content": prompt[1]["content"] + "\n\n" + explanation,
                },
            ]
        return self.cache_few_shot_prompts
