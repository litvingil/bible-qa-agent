"""
Test script to demonstrate query enhancement with pronouns and references.
This shows how the LLM-based enhance_query function resolves Hebrew pronouns
and contextual references from conversation history.
"""

import os
from dotenv import load_dotenv
from src.generator import ResponseGenerator
from src.search import BibleSearch

# Load environment variables
load_dotenv()

def test_query_enhancement():
    """Test query enhancement with example conversations."""
    
    # Initialize components
    bible_search = BibleSearch()
    generator = ResponseGenerator(bible_search, model="gpt-4o-mini")
    
    print("=== Query Enhancement Examples ===\n")
    
    # Example 1: Resolving "הוא" (he)
    print("Example 1: Resolving pronoun 'הוא' (he)")
    history1 = [
        {"role": "user", "content": "מי זה אברהם?"},
        {"role": "assistant", "content": "אברהם (שהתחיל נקרא אברם) הוא דמות מרכזית בספר בראשית: בן תרח, שעל־פי המסופר יצא עם משפחתו מאור כשדים ושהה בחרן לפני שה' קרא לו ללכת אל ארץ כנען[1][2]. ה' כרת עמו ברית, שינה את שמו לאברהם והבטיח שיהיה \"אב המון גוים\" וייתן לו ולזרעו את ארץ כנען[3]. אברהם נישא לשרה; בכהונה אלוהית נולד לו יצחק בהיותו בן מאה, והוא מילה את בנו כנדרש בברית[4]. אלוהים נסה את אמונתו בציווי להעלות את יצחק לעולה (עקדת יצחק), אך מלאך ה' עצר את המעשה ובמקומו נאסר איל כזבח[5]. אברהם קנה את מערת המכפלה בקניית קבר לשרה ושם נקברו שרה ואחר כך גם אברהם; על־פי המסופר נפטר בשיבה טובה ונקבר שם[6]. לו גם נולדו בנים אחרים — ישמעאל וילדי פילגשיו — אך את כל אשר לו נתן ליצחק, ויש תולדות ישמעאל[7].\n\n───\n[1] בראשית יא:כז-לב (עמוד 13)  \n[2] בראשית יב:א-ט (עמוד 14)  \n[3] בראשית יז:א-ח (עמוד 18)  \n[4] בראשית כא:א-ה, לא-לד (עמוד 23-24)  \n[5] בראשית כב:א-יד (עמוד 25)  \n[6] בראשית כג:א-כ (עמוד 26); בראשית כה:ז-י (עמוד 30)  \n[7] בראשית כה:ה-ו, יב (עמוד 30)"}
    ]
    query1 = "מה הוא עשה אחר כך?"
    enhanced1 = generator.enhance_query(query1, history1)
    print(f"  Original: {query1}")
    print(f"  Enhanced: {enhanced1}")
    print()
    query1 = "'מה זה ההבטחה הזאת שהובטחה? בין מי למי זה?'"
    enhanced1 = generator.enhance_query(query1, history1)
    print(f"  Original: {query1}")
    print(f"  Enhanced: {enhanced1}")
    print()
    history1.append({"role": "user", "content": "איפה מסופר שהוא יוצא לדרך?"})
    history1.append({"role": "assistant", "content": "היציאה מתוארת בשני מקומות עיקריים: בראשית יא:לא-לב — תרח לקח את אברם ויצאו מאור כשדים ויבאו עד־חרן[1] — ובבראשית יב:א-ו, שם מופיע ציווי ה' ל\"אברם: לך־לך מארצך...\" וכתוצאה \"וילך אברם... ויצאו ללכת ארצה כנען\"[2].\n\n\"וַיֹּאמֶר יְהוָה אֶל־אַבְרָם לֶךְ־לְךָ מֵאַרְצְךָ...\"[2]\n\n───\n[1] בראשית יא:לא-לב (עמוד 13)  \n[2] בראשית יב:א-ו (עמוד 14)"})
    query1 = "'מה זה ההבטחה הזאת שהובטחה? בין מי למי זה?'"
    enhanced1 = generator.enhance_query(query1, history1)
    print(f"  Original: {query1}")
    print(f"  Enhanced: {enhanced1}")
    print()
    
    # Example 2: Resolving "לו" (to him)
    print("Example 2: Resolving pronoun 'לו' (to him)")
    history2 = [
        {"role": "user", "content": "מי זה יעקב?"},
        {"role": "assistant", "content": "יעקב הוא בנו של יצחק ונכדו של אברהם..."},
        {"role": "user", "content": "ספר לי על החלום שלו"},
        {"role": "assistant", "content": "בחלום ראה יעקב סולם מוצב ארצה..."}
    ]
    query2 = "ומה אלוהים הבטיח לו?"
    enhanced2 = generator.enhance_query(query2, history2)
    print(f"  Original: {query2}")
    print(f"  Enhanced: {enhanced2}")
    print()
    
    # Example 3: Resolving "הפרק הקודם" (previous chapter)
    print("Example 3: Resolving 'הפרק הקודם' (previous chapter)")
    history3 = [
        {"role": "user", "content": "מה קרה בבראשית פרק 5?"},
        {"role": "assistant", "content": "פרק 5 מתאר את תולדות אדם..."}
    ]
    query3 = "מה קרה בפרק הקודם?"
    enhanced3 = generator.enhance_query(query3, history3)
    print(f"  Original: {query3}")
    print(f"  Enhanced: {enhanced3}")
    print()
    
    # Example 4: Resolving "מה שהזכרת" (what you mentioned)
    print("Example 4: Resolving 'מה שהזכרת' (what you mentioned)")
    history4 = [
        {"role": "user", "content": "ספר לי על ברית אברהם"},
        {"role": "assistant", "content": "ברית אברהם היא ברית בין אלוהים לאברהם, שבה הובטח לו הארץ וצאצאים רבים..."}
    ]
    query4 = "השווה את מה שהזכרת לברית נח"
    enhanced4 = generator.enhance_query(query4, history4)
    print(f"  Original: {query4}")
    print(f"  Enhanced: {enhanced4}")
    print()
    
    # Example 5: Multiple pronouns
    print("Example 5: Resolving multiple pronouns")
    history5 = [
        {"role": "user", "content": "מי זה משה?"},
        {"role": "assistant", "content": "משה הוא מנהיג עם ישראל שהוציא אותם ממצרים..."},
        {"role": "user", "content": "ספר לי על הפעם הראשונה שאלוהים מדבר אליו"},
        {"role": "assistant", "content": "אלוהים מדבר למשה מתוך הסנה הבוער בהר חורב..."}
    ]
    query5 = "מה הוא אמר לו?"
    enhanced5 = generator.enhance_query(query5, history5)
    print(f"  Original: {query5}")
    print(f"  Enhanced: {enhanced5}")
    print()
    
    # Example 6: Context with "שם" (there)
    print("Example 6: Resolving 'שם' (there)")
    history6 = [
        {"role": "user", "content": "ספר לי על יוסף"},
        {"role": "assistant", "content": "יוסף היה בנו האהוב של יעקב, נמכר למצרים על ידי אחיו..."}
    ]
    query6 = "מה עוד קרה לו שם?"
    enhanced6 = generator.enhance_query(query6, history6)
    print(f"  Original: {query6}")
    print(f"  Enhanced: {enhanced6}")
    print()
    
    # Example 7: No enhancement needed
    print("Example 7: No pronouns or references - should stay the same")
    history7 = [
        {"role": "user", "content": "מי זה דוד?"}
    ]
    query7 = "ספר לי על שלמה המלך"
    enhanced7 = generator.enhance_query(query7, history7)
    print(f"  Original: {query7}")
    print(f"  Enhanced: {enhanced7}")
    print(f"  Same as original: {query7 == enhanced7}")
    print()
    
    # Example 8: Empty history
    print("Example 8: Empty history - should return original")
    query8 = "מה הוא עשה?"
    enhanced8 = generator.enhance_query(query8, [])
    print(f"  Original: {query8}")
    print(f"  Enhanced: {enhanced8}")
    print(f"  Same as original: {query8 == enhanced8}")
    print()

if __name__ == "__main__":
    test_query_enhancement()
