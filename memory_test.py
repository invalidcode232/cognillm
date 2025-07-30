#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
import tqdm

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.lib.base.memory.summary import SummaryBasedMemory

ENDPOINT: str = os.getenv("OPENAI_API_ENDPOINT")
DEPLOYMENT: str = os.getenv("OPENAI_API_DEPLOYMENT_NAME")
API_KEY: str = os.getenv("OPENAI_API_KEY")
API_VERSION: str = os.getenv("OPENAI_API_VERSION")

SUMMARY_WINDOW_SIZE: int = 5
PROFILE_PATH: str = "profiles/MadamChan"

summary_memory = SummaryBasedMemory(
    profile_path=PROFILE_PATH,
    summary_window_size=SUMMARY_WINDOW_SIZE,
    endpoint=ENDPOINT,
    deployment=DEPLOYMENT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

def test_memory(TEST_MESSAGES):
    """
    Test the memory with a series of messages.
    """
    for idx in tqdm.tqdm(range(0, len(TEST_MESSAGES), 2), desc="Processing messages"):
        user_message = {
            "role": "user",
            "content": TEST_MESSAGES[idx]
        }
        # For assistant messages, turn dict into a string
        assistant_message = {
            "role": "assistant",
            "content": str(TEST_MESSAGES[idx + 1]) if idx + 1 <= len(TEST_MESSAGES) else ""
        }
        summary_memory.update([user_message, assistant_message])

    # Retrieve the summary
    summary = summary_memory.get_summary_list()
    print("-"* 40)
    print("▶ Memory Summary:")
    for idx, entry in enumerate(summary):
        print(f"▶ Round {idx * SUMMARY_WINDOW_SIZE + 1} to {idx * SUMMARY_WINDOW_SIZE + SUMMARY_WINDOW_SIZE}:")
        print(entry,"\n")
    print("-"* 40)

TEST_MESSAGES = [
    # Pair 1: Opening and establishing rapport
    "Good morning, Madam Chan. Thank you for coming in today. How are you feeling about being here?",
    {
        "chain_of_thought": "They're asking how I feel about being here. This is exactly what I was dreading - having to talk about my feelings with a stranger. I should just say I'm fine and maybe they'll move on quickly. My sister said this would help, but I'm not sure. I feel exposed already and we haven't even started talking about real problems.",
        "automatic_thoughts": "I shouldn't be here. They're going to judge me. I need to keep this surface-level.",
        "emotions": ["Nervous", "Guarded", "Reluctant"],
        "openness": "Very closed off, will give minimal responses to avoid deeper discussion. Any personal details will be heavily filtered.",
        "behaviors": ["Gives brief, polite responses", "Avoids eye contact", "Mentions sister as external validation for being here"],
        "message": "Good morning. I'm... I'm fine, thank you. A bit nervous, I suppose. I don't usually talk about personal things with strangers. But my sister said it might help."
    },
    
    # Pair 2: Exploring current emotional state
    "I can understand feeling nervous - that's completely normal. Can you tell me a bit about what's been on your mind lately? What brought you to seek support?",
    {
        "chain_of_thought": "They said it's normal to feel nervous. That's... actually reassuring. They're not pushing me about being 'fine' which I expected. Maybe I can share a little bit about feeling overwhelmed - that's general enough. But I won't mention the marriage problems or how angry I am at my husband. I'll focus on the practical stuff - everyone depending on me.",
        "automatic_thoughts": "Everyone does depend on me. I really am drowning. Maybe they can understand the practical burden without me having to reveal the emotional mess.",
        "emotions": ["Overwhelmed", "Tired", "Slightly relieved"],
        "openness": "Cautiously willing to share surface-level struggles, particularly practical burdens. Will avoid emotional depth or marital specifics.",
        "behaviors": ["Speaks more freely about practical concerns", "Uses metaphor ('drowning') to express feeling", "Lists responsibilities to justify overwhelm"],
        "message": "It's just... everything feels so overwhelming. I wake up every morning and there's so much to do, so many people depending on me. My youngest daughter needs constant attention, my other children need help with school, my husband... well, that's another story. Sometimes I feel like I'm drowning."
    },
    
    # Pair 3: Digging deeper into family dynamics
    "It sounds like you're carrying a very heavy load. You mentioned your husband - can you tell me more about what's happening in your relationship?",
    {
        "chain_of_thought": "They picked up on the 'that's another story' comment about my husband. I should have known they'd ask about that. Now I'm stuck - I can't just say 'nothing' because I already hinted there was a problem. Maybe I can talk about the practical issues - the money situation - without getting into how angry and hurt I feel. The financial stress is a real problem that anyone could understand.",
        "automatic_thoughts": "I've said too much already. But the money issue is real - that's not just my emotions, that's facts. Maybe they can understand why I'm stressed without me having to admit how lonely I feel.",
        "emotions": ["Frustrated", "Angry", "Vulnerable"],
        "openness": "Reluctantly willing to discuss concrete marital problems (finances) but will avoid emotional aspects like loneliness, hurt, or feeling unloved.",
        "behaviors": ["Provides specific examples to validate complaints", "Focuses on husband's actions rather than her feelings", "Shows anger through tone but maintains control"],
        "message": "We used to be happy, you know, back in China. But since we came to Hong Kong, everything changed. He sends most of his money to his family back home - his parents, his siblings. Meanwhile, I'm working part-time just to pay for groceries and my sister to watch the baby. We fight about money constantly. And he barely acknowledges our youngest daughter exists."
    },
    
    # Pair 4: Exploring financial stress and its impact
    "Financial stress can put enormous pressure on a relationship. How do you cope when these arguments happen?",
    {
        "chain_of_thought": "They're validating that financial stress is hard on relationships. That feels good - they're not judging me for having problems. They're asking how I cope... I could tell them about keeping quiet and just focusing on work and children. That's what I do - I withdraw and just keep busy. Maybe that's not healthy, but it's how I survive.",
        "automatic_thoughts": "What's the point of fighting with him? He's made his choice. I just have to deal with it. At least work gives me a break from thinking about it.",
        "emotions": ["Resigned", "Exhausted", "Defeated"],
        "openness": "More willing to share coping mechanisms, especially practical ones. Beginning to show some self-awareness about whether her strategies are working.",
        "behaviors": ["Admits to avoidance behavior", "Shows pattern of withdrawal during conflict", "Reveals work as emotional escape"],
        "message": "I just... I keep quiet mostly. What's the point of fighting? He's made his priorities clear. I just focus on work and the children. Work helps me not think about it. But lately, even that's not enough. I come home exhausted, and there's still dinner to cook, homework to check, baby to feed... I feel like I'm failing everyone."
    },
    
    # Pair 5: Addressing guilt and self-blame
    "I hear you saying you feel like you're failing everyone. That sounds like you're putting a lot of pressure on yourself. Do you think it's realistic to expect yourself to handle everything perfectly?",
    {
        "chain_of_thought": "They're questioning whether I should expect myself to handle everything perfectly. That's... challenging. My automatic response is 'but if I don't do it, who will?' But they're making me think about whether that's fair to myself. I've never really questioned whether my expectations are realistic. I just know that everyone needs me to be strong.",
        "automatic_thoughts": "But if I don't do everything, who will? I have to be strong for everyone. That's what mothers do. That's what daughters do.",
        "emotions": ["Guilty", "Pressured", "Conflicted"],
        "openness": "Beginning to engage with challenging questions about her role and expectations. Shows some defensive reactions but also genuine consideration of the social worker's perspective.",
        "behaviors": ["Justifies her responsibilities", "Shows core belief about needing to be strong", "Reveals guilt about parents in China"],
        "message": "But if I don't do it, who will? My husband is too busy sending money to China to help with our own children. My parents are getting old in China, and my youngest brother doesn't take good care of them. I should be there for them too, but I can't even manage my own family properly. I'm supposed to be strong for everyone."
    },
    
    # Pair 6: Exploring support systems and isolation
    "You mentioned your sister earlier. Do you have other people you can talk to about these feelings?",
    {
        "chain_of_thought": "They're asking about support systems. I want to say my sister, but I don't really talk to her about the deep stuff - I pay her for childcare but I don't want to burden her with my problems. And work colleagues... absolutely not. People would gossip. I have this belief that if people knew how much I was struggling, they'd think less of me. Maybe I should examine that belief, but it feels so true.",
        "automatic_thoughts": "If people knew how much I was struggling, they would gossip and judge me. I have to keep up appearances. No one really understands anyway.",
        "emotions": ["Lonely", "Ashamed", "Isolated"],
        "openness": "Willing to discuss her isolation patterns and the beliefs that keep her from seeking support. Shows awareness of her own reasoning.",
        "behaviors": ["Explains avoidance of emotional intimacy", "Shows concern about others' judgment", "Maintains transactional relationship with sister"],
        "message": "My sister helps with childcare, but I pay her for that. I don't want to burden her with my problems - her family has their own financial struggles. At work, my colleagues are nice, but I would never tell them about my marriage. People gossip, you know? They would think less of me. I have to keep up appearances. Nobody really understands what it's like."
    },
    
    # Pair 7: Addressing cognitive patterns and beliefs
    "It sounds like you believe that showing your struggles would make others think less of you. Have you ever tested this belief? Has anyone ever reacted negatively when you've shared difficulties?",
    {
        "chain_of_thought": "They're directly challenging my belief about showing weakness. This is uncomfortable because... well, I've never actually tested it. I just assume people would react badly because that's what I learned growing up. My mother always said to be strong and handle things myself. But now they're asking if I've ever actually experienced someone reacting negatively... and I haven't. This is making me question something I've never questioned before.",
        "automatic_thoughts": "I've never actually tried sharing struggles with anyone. But that's how it works, right? If you show weakness, people lose respect. But... maybe I don't actually know that for sure.",
        "emotions": ["Uncertain", "Defensive", "Curious"],
        "openness": "Engaging more deeply with self-examination. Showing some willingness to question long-held beliefs, though still defensive about them.",
        "behaviors": ["Admits she's never tested her beliefs", "References childhood learning", "Shows some self-doubt about her assumptions"],
        "message": "Well... no, I suppose I've never really tried. But that's how it works, isn't it? If you show weakness, people lose respect for you. If you can't handle your own problems, you're just a burden to others. My mother always taught me to be strong, to handle things myself. But sometimes I wonder... maybe I'm not as strong as I thought I was."
    },
    
    # Pair 8: Exploring emotional regulation and coping
    "When you feel overwhelmed like this, what happens to your emotions? How do you manage when everything feels like too much?",
    {
        "chain_of_thought": "They want to know about my emotions when I'm overwhelmed. This is getting very personal. I could tell them about crying at night - that's something I've never told anyone. It feels scary to admit that I cry, because that feels like admitting weakness. But also... maybe it would be a relief to say it out loud. They seem understanding so far.",
        "automatic_thoughts": "I shouldn't admit that I cry. That's showing weakness. But I'm so tired of keeping it all inside. Maybe it's okay to be human.",
        "emotions": ["Sad", "Lonely", "Guilty", "Slightly relieved"],
        "openness": "Significantly more open, willing to share vulnerable emotional experiences. Shows internal conflict about expressing emotions.",
        "behaviors": ["Admits to emotional suppression during day", "Reveals private crying at night", "Shows guilt about having feelings"],
        "message": "I try not to think about it. I just keep busy - work, cleaning, cooking, taking care of the children. But sometimes, at night when everyone's asleep, I just... I cry. Quietly, so nobody hears. I feel so sad, so alone. And then I feel guilty for feeling sad when I should be grateful. We have a roof over our heads, the children are healthy... I should be stronger than this."
    },
    
    # Pair 9: Addressing relationship with children and parenting concerns
    "You care so deeply about your children. How do you think all this stress affects your relationship with them?",
    {
        "chain_of_thought": "This question hits right at my biggest fear - that I'm failing my children. They're asking about how my stress affects them, which is what keeps me up at night. I know they can sense the tension. My oldest asks questions about why daddy and I don't talk. And I'm angry that my husband ignores our youngest, but I don't know how to fix it. This is my deepest shame - that I might not be giving them what they need.",
        "automatic_thoughts": "I'm failing them. They can see that something's wrong. I'm not giving them the stable home they deserve. This is all my fault.",
        "emotions": ["Anxious", "Guilty", "Angry", "Protective"],
        "openness": "Very open about parenting concerns and fears. This touches her core motivation - protecting and caring for her children.",
        "behaviors": ["Shows deep concern for children's wellbeing", "Admits children are aware of family stress", "Expresses anger toward husband's treatment of youngest"],
        "message": "That's what worries me most. I try to hide it from them, but children are smart. My oldest sometimes asks why Daddy and I don't talk much anymore. And my youngest... she's so sweet, but my husband acts like she doesn't exist. I'm angry about that, but I don't know how to fix it. I'm afraid I'm not giving them the stable home they deserve. They didn't ask for any of this."
    },
    
    # Pair 10: Moving toward hope and future planning
    "Despite all these challenges, you're still here, still fighting for your family. That takes tremendous strength. What would need to change for you to feel more supported and less overwhelmed?",
    {
        "chain_of_thought": "They're calling me strong for still fighting for my family. That's... not how I see myself. I see myself as failing, but they're reframing it as strength. That's a new perspective. They're asking what would need to change... I've never really thought about that. I just endure. But maybe I could imagine what support would look like. Maybe I could ask my sister for help without paying her. Maybe I could learn to be vulnerable without feeling like a failure.",
        "automatic_thoughts": "Maybe I am stronger than I think. Maybe fighting for my family, even when I feel like I'm failing, is actually strength. Maybe I could learn to accept help.",
        "emotions": ["Hopeful", "Uncertain", "Vulnerable", "Slightly empowered"],
        "openness": "Most open yet, engaging with future-focused thinking and considering new possibilities. Shows willingness to be vulnerable and consider change.",
        "behaviors": ["Accepts reframing of her situation", "Considers specific changes", "Shows openness to accepting help", "Acknowledges fear of vulnerability while considering it"],
        "message": "I... I never thought about it that way. Maybe you're right that I am stronger than I think. I suppose... I wish my husband and I could talk without fighting. I wish I didn't feel so alone in everything. And maybe... maybe I could learn to ask for help without feeling like a failure. My sister offered to help more, but I always insist on paying her. Maybe real support doesn't always have a price tag. But it's scary to be vulnerable, you know?"
    }
]

test_memory(TEST_MESSAGES)