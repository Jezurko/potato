#pragma once

#include <optional>

namespace potato::util {
    template< typename ...types >
    struct type_list;
    namespace detail {
        using empty_t_list = type_list<>;

        template< typename list >
        struct front;

        template<>
        struct front< empty_t_list >{
            using type = std::nullopt_t;
        };

        template< typename head, typename ...tail >
        struct front< type_list< head, tail... > >{
            using type = head;
        };

        template< typename type_list >
        struct pop_front;

        template< typename head, typename ...tail >
        struct pop_front< type_list< head, tail... > > {
            using type = type_list< tail... >;
        };

        template<>
        struct pop_front< empty_t_list > {
            using type = type_list< empty_t_list >;
        };

        template< typename ...lists >
        struct concat {};

        template< typename left, typename right >
        struct concat_list {};

        template< typename ...left, typename ...right >
        struct concat_list< type_list< left... >, type_list< right... > >
        {
            using type = type_list< left..., right... >;
        };

        template< typename list >
        struct concat< list >{
            using type = list;
        };

        template< typename list, typename ...rest >
        struct concat< list, rest... >{
            using type = typename concat_list< list, typename concat< rest... >::type >::type;

        };
    } // namespace detail

    template< typename ...types >
    struct type_list{
        using self = type_list;
        static constexpr std::size_t size = sizeof...(types);
        static constexpr bool empty = size == 0;
        using head = typename detail::front< self >::type;
        using tail = typename detail::pop_front< self >::type;
    };

    template< typename ...lists >
    using concat = typename detail::concat< lists... >::type;

} // namespace potato::util
