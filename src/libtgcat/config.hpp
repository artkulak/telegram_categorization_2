#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace Config
{

    namespace Language::Code
    {
        static constexpr auto English = "en";
        static constexpr auto Russian = "ru";
    } // Language::Code

    namespace Language::Model
    {
        static constexpr auto language = "../../models/sl_language/fasttext_language.ftz";
        static constexpr auto category_en = "../../models/en";
        static constexpr auto category_ru = "../../models/ru";
    } // Language::Model

    namespace Randomized
    {
        static constexpr auto posts_threshold = 10UL;
        static constexpr auto no_of_passes = 5UL;
    } // Randomized

} // Config

#endif // CONFIG_HPP
