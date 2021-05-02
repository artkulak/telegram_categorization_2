#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace Config
{

    namespace Language::Code
    {
        static constexpr auto English = "en";
        static constexpr auto Russian = "ru";
        static constexpr auto Uzbek = "uz";
        static constexpr auto Arabic = "ar";
        static constexpr auto Persian = "fa";
    } // Language::Code

    namespace Language::Model
    {
        static constexpr auto language = "../../models/sl_language/fasttext_language.ftz";
        static constexpr auto category_en = "../../models/en/fasttext_50_en.ftz";
        static constexpr auto category_ru = "../../models/ru/fasttext_50_ru.ftz";
        static constexpr auto category_uz = "../../models/uz/fasttext_50_uz.ftz";
        static constexpr auto category_ar = "../../models/ar/fasttext_50_ar.ftz";
        static constexpr auto category_fa = "../../models/fa/fasttext_50_fa.ftz";
    } // Language::Model

    namespace Randomized
    {
        static constexpr auto posts_threshold = 10UL;
        static constexpr auto no_of_passes = 5UL;
    } // Randomized

} // Config

#endif // CONFIG_HPP
